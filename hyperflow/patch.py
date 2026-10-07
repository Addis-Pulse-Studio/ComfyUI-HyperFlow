# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""Putting HyperFlow on a ComfyUI MiniMax H3 ModelPatcher.

Three things, all on a clone, all reversible by ComfyUI's own unpatching:

1. the 314 LoRA targets that exist natively (blocks, token refiner, base time embedder) go on either as ordinary
   weight patches through ``comfy.lora.load_lora`` -- q/k/v as row-slice patches of the fused ``qkv_proj`` -- or as
   forward hooks that never touch the weights (see :mod:`hyperflow.bypass` for why that matters on a quantized
   checkpoint, and :data:`APPLY_MODES` for how the choice is made);
2. an endpoint time embedder (the checkpoint's original time embedder plus the endpoint LoRA, merged, fp32) and an
   object patch on ``time_embedder.forward`` that blends the two embeddings;
3. a diffusion-model wrapper that, per forward, works out each distinct timestep's endpoint from the sampler's sigma
   schedule -- the part ComfyUI's sampler loop never passes on its own.
"""

from __future__ import annotations

import copy
import logging
import types

import torch

from . import bypass as hf_bypass
from . import keys as hf_keys
from .bypass import INJECTION_KEY  # re-exported: the key the bypass hooks are installed under
from .embedder import HyperFlowController, StepContext
from .header import read_metadata
from .schedule import AUDIO_SHIFT, DEFAULT_SIGMAS_8STEP, VIDEO_SHIFT, is_subgrid, shift_sigmas, validate_sigmas

logger = logging.getLogger("HyperFlow")

WRAPPER_KEY = "hyperflow"
OPTIONS_KEY = "hyperflow"
#: How the LoRA goes on. "patch" merges it into the weights: exact and free per step on a bf16 checkpoint, but on a
#: quantized one ComfyUI dequantizes, merges, then requantizes all 314 weights (see :mod:`hyperflow.bypass`).
#: "bypass" leaves the weights bit-identical and adds up(down(x)) in each forward instead. "auto" merges when no
#: target is quantized and bypasses when any is -- whole-model, not per target: on an H3 int8 checkpoint every
#: block Linear is quantized anyway, and a per-target split would double the test surface for nothing.
APPLY_MODES = ("auto", "patch", "bypass")
#: At sigma = 1 the video and audio timesteps are both 0 and ComfyUI gives them one embedding row, but their
#: endpoints differ. Evaluating that step at sigma * (1 - STEP0_NUDGE) separates the rows (t_video ~ 1e-4,
#: t_audio ~ 4e-4); the sinusoidal input moves by < 1e-3 rad, far below the step's own resolution.
STEP0_NUDGE = 1e-4
_SIGMA_MATCH_TOL = 1e-3
#: Non-pruned H3 checkpoints this pack is verified on, named in the refusal so a wrong pick is actionable.
SUPPORTED_CHECKPOINTS = ("minimax_h3_{fl2va,ref2va}_int8_convrot", "minimax_h3_{fl2va,ref2va}_bf16",
                         "Minimax-h3_Singularity_ref2va_v1.3_int8")


def check_model(model) -> torch.nn.Module:
    """Return the H3 diffusion model, or raise with what to load instead."""
    import comfy.model_base

    base = getattr(model, "model", None)
    if not isinstance(base, comfy.model_base.MiniMaxH3):
        raise ValueError(f"HyperFlow is a MiniMax H3 adapter; this model is {type(base).__name__}.")
    dm = base.diffusion_model
    if getattr(dm, "use_adaln_curves", False) or not hasattr(dm, "time_embedder"):
        raise ValueError(
            "This MiniMax H3 checkpoint is the curve-form variant (a baked adaln_t_table, no time_embedder), e.g. "
            "the *_pruned_* files. HyperFlow conditions every step on the interval it integrates through a second "
            "time embedder, so it needs a checkpoint that still has time_embedder.* weights. Works: "
            + ", ".join(SUPPORTED_CHECKPOINTS) + "."
        )
    import comfy.patcher_extension

    if model.wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {}).get(WRAPPER_KEY):
        raise ValueError("This model already carries HyperFlow; load it once, on the plain base model.")
    return dm


def build_endpoint_module(model, dm, endpoint_lora, alpha, rank, strength):
    """Original (unpatched) time embedder weights + the endpoint LoRA, merged into a standalone fp32 module."""
    module = copy.deepcopy(dm.time_embedder).to("cpu", torch.float32)
    prefix = "diffusion_model.time_embedder."
    original = model.get_key_patches(prefix)  # backup-aware: the checkpoint's weights, never a patched copy
    scale = strength * alpha / rank
    for name, param in module.named_parameters():
        entry = original.get(prefix + name)
        if entry is None:
            raise KeyError(f"HyperFlow: this model has no {prefix}{name}; it is not a MiniMax H3 time embedder "
                           "this pack knows how to copy for the step endpoint.")
        # [0] is the model's own weight paired with the op's converter; applying it dequantizes a quantized weight
        # instead of reading its packed storage as numbers.
        weight, convert = entry[0]
        weight = convert(weight.detach().to("cpu", copy=True)).to(torch.float32)
        leaf = name.split(".")[0]
        if name.endswith(".weight") and leaf in endpoint_lora:
            a, b = endpoint_lora[leaf]
            delta = b.to(torch.float32) @ a.to(torch.float32)
            if delta.shape != weight.shape:
                raise ValueError(f"endpoint {leaf}: LoRA delta {tuple(delta.shape)} vs weight {tuple(weight.shape)}.")
            weight = weight + scale * delta
        param.data = weight
    # The model's operations may be ComfyUI cast ops; the copy runs on its own device at fp32 either way.
    for sub in module.modules():
        if hasattr(sub, "comfy_cast_weights"):
            sub.comfy_cast_weights = False
    return module


def resolve_mode(model, mode: str, patches) -> tuple[str, int]:
    """``(patch | bypass, how many targets are quantized)``. Only ``auto`` looks at the checkpoint."""
    if mode not in APPLY_MODES:
        raise ValueError(f"HyperFlow apply mode must be one of {APPLY_MODES}, got {mode!r}.")
    quantized = hf_bypass.quantized_count(model, patches)
    if mode != "auto":
        return mode, quantized
    return ("bypass" if quantized else "patch"), quantized


def apply_hyperflow(model, path: str, strength: float = 1.0, mode: str = "auto", gate=None, sigmas=None):
    """Return a clone of ``model`` with HyperFlow applied. ``model`` is a ComfyUI ModelPatcher.

    ``gate`` and ``sigmas`` override the weights file's own header; both default to it.
    """
    import comfy.lora
    import comfy.patcher_extension
    import comfy.utils

    if mode not in APPLY_MODES:  # before loading 3.6 GB of weights
        raise ValueError(f"HyperFlow apply mode must be one of {APPLY_MODES}, got {mode!r}.")
    dm = check_model(model)
    metadata = read_metadata(path)
    # pre-converted files (native names, fused q|k|v) are unpacked to Video Rebirth's layout first
    sd = hf_keys.normalize(comfy.utils.load_torch_file(path, safe_load=True), metadata.raw)

    heads, head_dim = dm.blocks[0].attn.heads, dm.blocks[0].attn.head_dim
    converted = hf_keys.convert(sd, metadata.lora_alpha, qkv_rows=heads * head_dim)
    if metadata.lora_rank is not None and metadata.lora_rank != converted.rank:
        raise ValueError(f"Header says lora_rank={metadata.lora_rank} but the tensors have rank {converted.rank}.")
    _check_shapes(model, converted)
    gate = metadata.gate if gate is None else float(gate)
    raw_sigmas = list(metadata.sigmas or DEFAULT_SIGMAS_8STEP) if not sigmas else validate_sigmas(sigmas)

    m = model.clone()
    patches = comfy.lora.load_lora(converted.lora_sd, converted.key_map, log_missing=False)
    if len(patches) != len(converted.key_map):
        raise ValueError(f"ComfyUI built {len(patches)} LoRA patches for {len(converted.key_map)} HyperFlow targets.")
    resolved, quantized = resolve_mode(model, mode, patches)
    merged_in_bypass = {}
    if resolved == "bypass":
        merged_in_bypass = hf_bypass.install(m, patches, strength)
        if merged_in_bypass:
            logger.info("HyperFlow bypass: %d target(s) via merge (ComfyUI could not hook them: %s).",
                        len(merged_in_bypass),
                        ", ".join(sorted(str(k) for k in merged_in_bypass)[:3]))
        to_patch = merged_in_bypass
    else:
        to_patch = patches
    if to_patch:
        patched = m.add_patches(to_patch, strength)
        if len(patched) != len(to_patch):
            missing = sorted(str(k) for k in set(to_patch) - set(patched))
            raise ValueError(f"{len(missing)} HyperFlow patch target(s) are not in this model, e.g. {missing[:4]}.")

    endpoint = build_endpoint_module(m, dm, converted.endpoint, metadata.lora_alpha, converted.rank, strength)
    controller = HyperFlowController(endpoint, gate, metadata)
    base_forward = types.MethodType(type(dm.time_embedder).forward, dm.time_embedder)
    m.add_object_patch("diffusion_model.time_embedder.forward", controller.make_forward(base_forward))

    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, make_wrapper(controller, dm, raw_sigmas)
    )
    m.model_options[OPTIONS_KEY] = {
        "sigmas": raw_sigmas,
        "video_shift": metadata.video_shift or VIDEO_SHIFT,
        "audio_shift": metadata.audio_shift or AUDIO_SHIFT,
        "version": metadata.version,
        "gate": gate,
        "apply_mode": resolved,
    }
    how = "bypass LoRAs" if resolved == "bypass" else "weight patches"
    if resolved == "bypass" and merged_in_bypass:
        how = f"bypass LoRAs ({len(merged_in_bypass)} via merge)"
    logger.info(
        "HyperFlow %s: %s%d %s + two-time embedder (gate %.4g, rank %d, strength %.3g).",
        metadata.version,
        f"auto -> {resolved} ({quantized} quantized targets), " if mode == "auto" else "",
        len(patches), how, gate, converted.rank, strength,
    )
    return m


def _check_shapes(model, converted) -> None:
    for module, target in converted.targets.items():
        if target.endpoint is not None:
            continue
        spec = converted.key_map[module]
        key, rows = (spec, None) if isinstance(spec, str) else (spec[0], spec[1][2])
        layer = hf_bypass.module_of(model, key)
        if layer is None:
            raise KeyError(f"HyperFlow target {key} ({module}) does not exist on this model.")
        # from the module, not its stored weight: a quantized weight's shape is packed storage, not geometry
        out_rows, in_cols = hf_bypass.out_features(layer, key), hf_bypass.in_features(layer, key)
        a = converted.lora_sd[f"{module}.lora_A.weight"]
        b = converted.lora_sd[f"{module}.lora_B.weight"]
        expect_rows = rows if rows is not None else out_rows
        if target.qkv_index is not None and out_rows != 3 * rows:
            raise ValueError(f"{key}: {out_rows} rows is not 3 x {rows}; not a fused q|k|v projection.")
        if a.shape[1] != in_cols or b.shape[0] != expect_rows:
            raise ValueError(
                f"Shape mismatch for {module} -> {key}: Linear {in_cols} -> {out_rows}, "
                f"lora_A {tuple(a.shape)}, lora_B {tuple(b.shape)}."
            )


def make_wrapper(controller: HyperFlowController, dm, raw_sigmas):
    import comfy.ldm.minimax.model as h3

    state = {"checked": None, "warned_cfg": False}

    def wrapper(executor, x, timestep, context, transformer_options, *args, **kwargs):
        sample_sigmas = transformer_options.get("sample_sigmas")
        if sample_sigmas is None:
            raise RuntimeError("HyperFlow needs the sampler's sigma schedule (transformer_options['sample_sigmas']).")
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", dm.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", dm.sigma_shift_audio))

        schedule = [float(s) for s in sample_sigmas.flatten().tolist()]
        signature = (tuple(schedule), shift_v)
        if state["checked"] != signature:
            grid = shift_sigmas(raw_sigmas, shift_v)
            if not is_subgrid(schedule, grid):
                raise ValueError(
                    "HyperFlow runs a fixed 8-step grid stored in its weights file, but this sampler was given "
                    f"{len(schedule) - 1} steps on another schedule. Feed the sampler's sigmas from the "
                    "'HyperFlow Sigmas' node (euler, cfg 1.0); a different schedule would silently condition every "
                    "step on the wrong interval."
                )
            state["checked"] = signature

        sigma = float((timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6))
        index = min(range(len(schedule)), key=lambda i: abs(schedule[i] - sigma))
        if abs(schedule[index] - sigma) > _SIGMA_MATCH_TOL or index + 1 >= len(schedule):
            raise ValueError(
                f"HyperFlow got a model call at sigma {sigma:.5f}, which is not a step of its grid. Use the 'euler' "
                "sampler: multi-stage samplers (heun, dpm_2, ...) evaluate between grid points."
            )
        sigma_next = schedule[index + 1]

        if sigma >= 1.0 - 1e-6:
            timestep = timestep * (1.0 - STEP0_NUDGE)
        # exactly as MiniMaxH3Model._forward derives them, so the float32 values match its unique_t
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - h3.time_shift_sigma(sigma_v, shift_v, shift_a))
        r_v = 1.0 - sigma_next
        r_a = 1.0 - float(h3.time_shift_sigma(torch.tensor(sigma_next, dtype=torch.float32), shift_v, shift_a))

        payload = kwargs.get("minimax_payload") or {}
        pin = min(
            float(payload.get("visual_cond_noise_aug", h3.VISUAL_COND_TIMESTEP)),
            float(payload.get("audio_cond_noise_aug", h3.AUDIO_COND_TIMESTEP)),
            h3.VISUAL_COND_TIMESTEP,
        )
        if 1 in (transformer_options.get("cond_or_uncond") or []) and not state["warned_cfg"]:
            state["warned_cfg"] = True
            logger.warning("HyperFlow was distilled for cfg 1.0; this run evaluates an unconditional branch.")

        previous = controller.context
        controller.context = StepContext(t_v, t_a, r_v, r_a, pin, float(sigma_v), sigma_next)
        try:
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        finally:
            controller.context = previous

    return wrapper
