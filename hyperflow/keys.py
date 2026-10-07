# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""HyperFlow's diffusers/PEFT key names -> ComfyUI's native MiniMax H3 module paths.

The weights file stores ``transformer.<diffusers module>.lora_{A,B}.weight``. ComfyUI's H3 uses the checkpoint's own
names, which differ in four ways (the renames diffusers' ``convert_minimax_h3_to_diffusers.py`` applies, reversed):

* ``transformer_blocks.i`` -> ``blocks.i``; ``token_refiner.refiner_blocks.j`` -> ``token_refiner.blocks.j``;
* ``attn.to_q`` / ``to_k`` / ``to_v`` are contiguous row thirds of ComfyUI's fused ``attn.qkv_proj``, so each becomes
  a patch on a row slice of it; ``attn.to_out.0`` -> ``attn.out_proj``;
* ``ff.net.0.proj`` -> ``mlp.fc1`` with the two halves of its output swapped: diffusers' SwiGLU is ``[value; gate]``,
  ComfyUI's fused ``fc1`` is ``[gate; value]``, so ``lora_B``'s row halves swap; ``ff.net.2`` -> ``mlp.fc2``;
* ``time_embedder.linear_1`` / ``linear_2`` -> ``time_embedder.proj_in`` / ``proj_out``.

``endpoint_time_embedder.*`` has no native module: it is returned separately for the endpoint embedder.

Pre-converted files (e.g. ``drbaph/Hyperflow-Comfyui``) already use the native names, with q/k/v packed into one
block-diagonal LoRA on ``attn.qkv_proj`` (``A = [A_q; A_k; A_v]``, ``B = diag(B_q, B_k, B_v)``). ``to_diffusers``
unpacks such a file back into the layout above, losslessly, so both kinds of file go through the same ``convert``.

Everything here is pure: tensors in, tensors out, no ComfyUI import.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

KEY_RE = re.compile(r"^transformer\.(?P<module>.+)\.lora_(?P<matrix>[AB])\.weight$")

_BLOCK_PREFIXES = (
    (re.compile(r"^transformer_blocks\.(\d+)\.(.+)$"), "blocks.{}.{}"),
    (re.compile(r"^token_refiner\.refiner_blocks\.(\d+)\.(.+)$"), "token_refiner.blocks.{}.{}"),
)
_QKV = {"attn.to_q": 0, "attn.to_k": 1, "attn.to_v": 2}
_BLOCK_RENAMES = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}
_TIME_RENAMES = {"linear_1": "proj_in", "linear_2": "proj_out"}


@dataclass(frozen=True)
class Target:
    """Where one file module lands: a native ``.weight`` key, optionally a row slice ``(index, of_n)`` of it."""

    native: str
    qkv_index: int | None = None
    swap_halves: bool = False
    endpoint: str | None = None  # "proj_in" / "proj_out" for endpoint_time_embedder modules


def map_module(module: str) -> Target:
    """Map one diffusers module path (without the ``transformer.`` prefix) onto ComfyUI's H3."""
    for prefix in ("time_embedder.", "endpoint_time_embedder."):
        if module.startswith(prefix):
            leaf = module[len(prefix):]
            if leaf not in _TIME_RENAMES:
                break
            if prefix == "time_embedder.":
                return Target(native=f"time_embedder.{_TIME_RENAMES[leaf]}")
            return Target(native="", endpoint=_TIME_RENAMES[leaf])
    for pattern, template in _BLOCK_PREFIXES:
        match = pattern.match(module)
        if match is None:
            continue
        index, rest = match.groups()
        if rest in _QKV:
            return Target(native=template.format(index, "attn.qkv_proj"), qkv_index=_QKV[rest])
        if rest in _BLOCK_RENAMES:
            return Target(native=template.format(index, _BLOCK_RENAMES[rest]), swap_halves=rest == "ff.net.0.proj")
        break
    raise KeyError(
        f"HyperFlow module {module!r} has no counterpart in ComfyUI's MiniMax H3. The file is either not the "
        "HyperFlow MiniMax-H3 adapter or a newer layout this node does not know yet."
    )


def plan_from_keys(keys) -> dict[str, dict[str, str]]:
    """Group the file's keys by module: ``{module: {"A": key, "B": key}}``. Any other key is an error."""
    plan: dict[str, dict[str, str]] = {}
    for key in keys:
        match = KEY_RE.match(key)
        if match is None:
            raise ValueError(
                f"Unexpected key {key!r} in a HyperFlow weights file; only "
                "`transformer.<module>.lora_A.weight` / `lora_B.weight` keys are allowed."
            )
        plan.setdefault(match["module"], {})[match["matrix"]] = key
    incomplete = sorted(m for m, pair in plan.items() if set(pair) != {"A", "B"})
    if incomplete:
        raise ValueError(f"Modules with only one of lora_A / lora_B: {incomplete[:8]}")
    return plan


@dataclass
class Converted:
    """ComfyUI-ready pieces of a HyperFlow file."""

    #: LoRA tensors under the file's own module names (``{module}.lora_A.weight`` ...), fc1 ``lora_B`` halves swapped.
    lora_sd: dict = field(default_factory=dict)
    #: ``comfy.lora.load_lora`` key map: module name -> ``diffusion_model.<native>.weight`` or ``(key, (dim, start, n))``.
    key_map: dict = field(default_factory=dict)
    #: ``{"proj_in": (A, B), "proj_out": (A, B)}`` of the endpoint time embedder.
    endpoint: dict = field(default_factory=dict)
    #: ``{module: Target}``, for validation against the model.
    targets: dict = field(default_factory=dict)
    rank: int = 0


def convert(sd: dict, lora_alpha: float, qkv_rows: int | None = None, prefix: str = "diffusion_model.") -> Converted:
    """Convert a loaded HyperFlow state dict.

    Args:
        sd: ``{key: tensor}`` of the file.
        lora_alpha: From the header; stored as each module's ``.alpha`` so ComfyUI scales by ``alpha / rank``.
        qkv_rows: Rows of one of q/k/v in the fused ``qkv_proj`` (heads * head_dim). Defaults to ``lora_B``'s rows.
    """
    import torch

    plan = plan_from_keys(sd.keys())
    out = Converted()
    ranks = set()
    for module, pair in sorted(plan.items()):
        target = map_module(module)
        a, b = sd[pair["A"]], sd[pair["B"]]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"{module}: lora_A {tuple(a.shape)} and lora_B {tuple(b.shape)} do not pair up.")
        ranks.add(int(a.shape[0]))
        out.targets[module] = target
        if target.endpoint is not None:
            out.endpoint[target.endpoint] = (a, b)
            continue
        if target.swap_halves:
            if b.shape[0] % 2:
                raise ValueError(f"{module}: a SwiGLU projection needs an even row count, got {b.shape[0]}.")
            value, gate = b.chunk(2, dim=0)
            b = torch.cat([gate, value], dim=0).contiguous()
        out.lora_sd[f"{module}.lora_A.weight"] = a
        out.lora_sd[f"{module}.lora_B.weight"] = b
        out.lora_sd[f"{module}.alpha"] = torch.tensor(float(lora_alpha))
        key = f"{prefix}{target.native}.weight"
        if target.qkv_index is None:
            out.key_map[module] = key
        else:
            rows = int(qkv_rows if qkv_rows is not None else b.shape[0])
            if b.shape[0] != rows:
                raise ValueError(f"{module}: lora_B has {b.shape[0]} rows, the fused qkv slice has {rows}.")
            out.key_map[module] = (key, (0, target.qkv_index * rows, rows))
    if len(ranks) != 1:
        raise ValueError(f"All HyperFlow modules must share one rank, found {sorted(ranks)}.")
    out.rank = ranks.pop()
    if set(out.endpoint) != {"proj_in", "proj_out"}:
        raise ValueError(
            "The file has no complete endpoint_time_embedder (linear_1 and linear_2). Without it HyperFlow's "
            "two-time conditioning cannot be built; this is not a HyperFlow file or it is truncated."
        )
    return out


NATIVE_KEY_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<matrix>[AB])\.weight$")
#: Header key of pre-converted files saying which order ``mlp.fc1``'s ``lora_B`` rows are in. Absent means diffusers'
#: ``[value; gate]`` (verified tensor-for-tensor on drbaph's v1.0 file against Video Rebirth's); ``gate_value`` means
#: ComfyUI's ``[gate; value]``.
FC1_LAYOUT_KEY = "hyperflow_fc1_layout"
FC1_LAYOUTS = ("value_gate", "gate_value")

_NATIVE_PREFIXES = (
    (re.compile(r"^blocks\.(\d+)\.(.+)$"), "transformer_blocks.{}.{}"),
    (re.compile(r"^token_refiner\.blocks\.(\d+)\.(.+)$"), "token_refiner.refiner_blocks.{}.{}"),
)
_NATIVE_RENAMES = {native: diffusers for diffusers, native in _BLOCK_RENAMES.items()}
_NATIVE_TIME_RENAMES = {native: diffusers for diffusers, native in _TIME_RENAMES.items()}


def is_native(keys) -> bool:
    """True for a file already under ComfyUI's native H3 names (no ``transformer.`` prefix)."""
    keys = list(keys)
    return bool(keys) and not any(key.startswith("transformer.") for key in keys)


def native_to_diffusers(module: str) -> tuple[str, ...]:
    """The diffusers module(s) one native module stands for: three for a fused ``attn.qkv_proj``, else one."""
    for prefix in ("time_embedder.", "endpoint_time_embedder."):
        if module.startswith(prefix) and module[len(prefix):] in _NATIVE_TIME_RENAMES:
            return (prefix + _NATIVE_TIME_RENAMES[module[len(prefix):]],)
    for pattern, template in _NATIVE_PREFIXES:
        match = pattern.match(module)
        if match is None:
            continue
        index, rest = match.groups()
        if rest == "attn.qkv_proj":
            return tuple(template.format(index, name) for name in _QKV)
        if rest in _NATIVE_RENAMES:
            return (template.format(index, _NATIVE_RENAMES[rest]),)
        break
    raise KeyError(
        f"Module {module!r} of a pre-converted HyperFlow file has no counterpart in ComfyUI's MiniMax H3; the file "
        "is either not the HyperFlow MiniMax-H3 adapter or a newer layout this node does not know yet."
    )


def to_diffusers(sd: dict, fc1_layout: str | None = None) -> dict:
    """Turn a pre-converted (native-named) HyperFlow state dict back into Video Rebirth's diffusers layout.

    Fused q/k/v LoRAs are split into their three row blocks; that is only exact when ``B`` is block-diagonal, so any
    other fused LoRA is refused instead of approximated. ``fc1_layout`` is the file's ``hyperflow_fc1_layout``.
    """
    import torch

    layout = fc1_layout or "value_gate"
    if layout not in FC1_LAYOUTS:
        raise ValueError(f"Unknown {FC1_LAYOUT_KEY} {fc1_layout!r} in the header; expected one of {FC1_LAYOUTS}.")
    plan: dict[str, dict[str, str]] = {}
    for key in sd:
        match = NATIVE_KEY_RE.match(key)
        if match is None:
            raise ValueError(
                f"Unexpected key {key!r} in a pre-converted HyperFlow file; only "
                "`<module>.lora_A.weight` / `lora_B.weight` keys are allowed."
            )
        plan.setdefault(match["module"], {})[match["matrix"]] = key
    incomplete = sorted(m for m, pair in plan.items() if set(pair) != {"A", "B"})
    if incomplete:
        raise ValueError(f"Modules with only one of lora_A / lora_B: {incomplete[:8]}")
    if not any(m.startswith("endpoint_time_embedder.") for m in plan):
        raise ValueError(
            "This pre-converted HyperFlow file has no endpoint_time_embedder: it is the *_pruned variant, which runs "
            "single-time and off-recipe. This pack runs the full two-time recipe only; use "
            "custom_node_hyperflow_8step_v1.0_comfyui.safetensors (not _pruned) on a non-pruned H3 checkpoint."
        )

    out = {}
    for module, pair in sorted(plan.items()):
        a, b = sd[pair["A"]], sd[pair["B"]]
        names = native_to_diffusers(module)
        if len(names) == 1:
            if names[0].endswith("ff.net.0.proj") and layout == "gate_value":
                if b.shape[0] % 2:
                    raise ValueError(f"{module}: a SwiGLU projection needs an even row count, got {b.shape[0]}.")
                gate, value = b.chunk(2, dim=0)
                b = torch.cat([value, gate], dim=0)
            out[f"transformer.{names[0]}.lora_A.weight"] = a
            out[f"transformer.{names[0]}.lora_B.weight"] = b
            continue
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] % 3 or b.shape[0] % 3 or b.shape[1] != a.shape[0]:
            raise ValueError(
                f"{module}: lora_A {tuple(a.shape)} / lora_B {tuple(b.shape)} is not a q|k|v LoRA packed three ways."
            )
        rank, rows = a.shape[0] // 3, b.shape[0] // 3
        for i, name in enumerate(names):
            for j in range(3):
                block = b[i * rows:(i + 1) * rows, j * rank:(j + 1) * rank]
                if i != j and bool(block.any()):
                    raise ValueError(
                        f"{module}: the fused q|k|v lora_B is not block-diagonal, so it cannot be split into exact "
                        "q, k and v patches."
                    )
            out[f"transformer.{name}.lora_A.weight"] = a[i * rank:(i + 1) * rank]
            out[f"transformer.{name}.lora_B.weight"] = b[i * rows:(i + 1) * rows, i * rank:(i + 1) * rank]
    return out


def normalize(sd: dict, header: dict | None = None) -> dict:
    """Any supported HyperFlow state dict in Video Rebirth's diffusers layout, ready for ``convert``."""
    if not is_native(sd.keys()):
        return sd
    return to_diffusers(sd, (header or {}).get(FC1_LAYOUT_KEY))
