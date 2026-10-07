"""Shared fixtures: ComfyUI on sys.path, a tiny MiniMax H3 and a synthetic HyperFlow file for it."""

from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "")
if COMFYUI_PATH and os.path.isdir(COMFYUI_PATH) and COMFYUI_PATH not in sys.path:
    sys.path.insert(0, COMFYUI_PATH)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

try:
    import comfy.model_base  # noqa: F401
    HAVE_COMFY = True
except Exception:  # pragma: no cover
    HAVE_COMFY = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "torch not installed")
needs_comfy = unittest.skipUnless(HAVE_COMFY, "ComfyUI not importable; set COMFYUI_PATH")


def _int8_available() -> bool:
    """Whether comfy-kitchen can quantize and multiply TensorWiseINT8Layout weights on this machine."""
    if not HAVE_COMFY:
        return False
    try:
        import torch
        from comfy.ops import QuantizedTensor

        weight = QuantizedTensor.from_float(torch.randn(32, 16), "TensorWiseINT8Layout")
        import comfy.ops as ops

        ops.linear_input_act_(torch.randn(4, 16), weight, None)
        return True
    except Exception:  # pragma: no cover - platform dependent
        return False


HAVE_INT8 = _int8_available()
needs_int8 = unittest.skipUnless(HAVE_INT8, "comfy-kitchen INT8 is not usable on this machine")

#: The Linear modules HyperFlow patches, by module-path suffix.
LORA_TARGET_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def quantize_lora_targets(patcher) -> dict:
    """Make every HyperFlow-target Linear genuinely INT8, as an ``*_int8_convrot`` checkpoint is.

    Returns ``{module path: dequantized weight}`` so a reference model can be built on the same values. This is the
    configuration the fp32 tests cannot reach: an INT8 Linear folds ``mlp.fc2``'s SwiGLU into its own kernel.
    """
    import torch
    from comfy.ops import QuantizedTensor

    dequantized = {}
    for name, module in patcher.model.diffusion_model.named_modules():
        if not name.endswith(LORA_TARGET_SUFFIXES):
            continue
        quantized = QuantizedTensor.from_float(module.weight.data.float(), "TensorWiseINT8Layout")
        dequantized[name] = quantized.dequantize().clone()
        module.weight = torch.nn.Parameter(quantized, requires_grad=False)
        module.quant_format = "int8_tensorwise"
    if not dequantized:
        raise AssertionError("no HyperFlow target Linears found to quantize")
    return dequantized

TINY = dict(
    image_model="minimax_h3", hidden_size=128, num_layers=2, token_refiner_num_layers=2, num_attention_heads=1,
    attention_head_dim=128, ffn_hidden_size=64, text_dim=48, timestep_input_dim=16, time_embed_hidden_size=32,
    time_embed_dim=24,
)
RANK = 4
ALPHA = 8.0  # scale 2.0, so a missing alpha would show
GATE = 0.7
RAW_SIGMAS = [1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0]


def tiny_model(seed: int = 0, **overrides):
    """A ComfyUI ModelPatcher around a 2-layer MiniMax H3 with random weights, on CPU in fp32."""
    import torch
    import comfy.model_base
    import comfy.model_patcher
    import comfy.supported_models

    config = comfy.supported_models.MiniMaxH3({**TINY, **overrides})
    config.set_inference_dtype(torch.float32, None)
    model = comfy.model_base.MiniMaxH3(config, device="cpu")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.diffusion_model.named_parameters():
            if name.endswith("norm.weight") or ".norm" in name or name.endswith("_norm.weight"):
                param.copy_(1.0 + 0.05 * torch.randn(param.shape, generator=gen))
            else:
                param.copy_(0.1 * torch.randn(param.shape, generator=gen))
        dm = model.diffusion_model
        dm.rope.inv_freq.copy_(1.0 / (10000.0 ** (torch.arange(16, dtype=torch.float32) / 16)))
        if getattr(dm, "use_adaln_curves", False):
            dm.adaln_t_table.copy_(torch.randn(dm.adaln_t_table.shape, generator=gen))
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)


def diffusers_modules(num_layers=2, num_refiner=2, hidden=128, inner=128, ffn=64, freq=16, te_hidden=32, te_out=24):
    """``{diffusers module: (in_features, out_features)}`` for a HyperFlow file on the tiny model."""
    mods = {}
    for prefix, n in (("transformer_blocks", num_layers), ("token_refiner.refiner_blocks", num_refiner)):
        for i in range(n):
            p = f"{prefix}.{i}."
            mods[p + "attn.to_q"] = (hidden, inner)
            mods[p + "attn.to_k"] = (hidden, inner)
            mods[p + "attn.to_v"] = (hidden, inner)
            mods[p + "attn.to_out.0"] = (inner, hidden)
            mods[p + "ff.net.0.proj"] = (hidden, 2 * ffn)
            mods[p + "ff.net.2"] = (ffn, hidden)
    for prefix in ("time_embedder", "endpoint_time_embedder"):
        mods[prefix + ".linear_1"] = (freq, te_hidden)
        mods[prefix + ".linear_2"] = (te_hidden, te_out)
    return mods


def hyperflow_state_dict(seed: int = 1, rank: int = RANK, **dims):
    import torch

    gen = torch.Generator().manual_seed(seed)
    sd = {}
    for module, (fan_in, fan_out) in diffusers_modules(**dims).items():
        sd[f"transformer.{module}.lora_A.weight"] = 0.05 * torch.randn(rank, fan_in, generator=gen)
        sd[f"transformer.{module}.lora_B.weight"] = 0.05 * torch.randn(fan_out, rank, generator=gen)
    return sd


def native_state_dict(sd, fc1_layout="value_gate"):
    """``sd`` as a pre-converted (drbaph-style) file: native names, q|k|v fused block-diagonally."""
    import torch

    renames = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2",
               "linear_1": "proj_in", "linear_2": "proj_out"}
    out, qkv = {}, {}
    for key, tensor in sd.items():
        module, matrix = key[len("transformer."):].rsplit(".lora_", 1)
        module = module.replace("transformer_blocks.", "blocks.").replace("refiner_blocks.", "blocks.")
        head, leaf = module.rsplit(".", 1) if module.endswith(("to_q", "to_k", "to_v")) else (module, None)
        if leaf is not None:
            qkv.setdefault(head, {}).setdefault(matrix, {})[leaf] = tensor
            continue
        for old, new in renames.items():
            if module.endswith("." + old):
                module = module[: -len(old)] + new
                break
        if module.endswith("mlp.fc1") and matrix.startswith("B") and fc1_layout == "gate_value":
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat([gate, value], dim=0)
        out[f"{module}.lora_{matrix}"] = tensor
    for head, parts in qkv.items():
        order = ("to_q", "to_k", "to_v")
        out[f"{head}.qkv_proj.lora_A.weight"] = torch.cat([parts["A.weight"][n] for n in order])
        out[f"{head}.qkv_proj.lora_B.weight"] = torch.block_diag(*[parts["B.weight"][n] for n in order])
    return out


def hyperflow_metadata(**overrides):
    meta = {
        "hyperflow": "true", "hyperflow_version": "1.0-test", "hyperflow_gate": str(GATE), "lora_alpha": str(ALPHA),
        "lora_rank": str(RANK), "base_model": "MiniMaxAI/MiniMax-H3", "hyperflow_sigmas": json.dumps(RAW_SIGMAS),
        "hyperflow_video_shift": "12.0", "hyperflow_audio_shift": "3.0", "tasks": json.dumps(["t2va", "fl2va", "ref2va"]),
    }
    meta.update(overrides)
    return meta


def write_hyperflow_file(directory: str, sd=None, metadata=None, name="hyperflow_test.safetensors") -> str:
    from safetensors.torch import save_file

    path = os.path.join(directory, name)
    save_file(sd if sd is not None else hyperflow_state_dict(), path,
              metadata=metadata if metadata is not None else hyperflow_metadata())
    return path
