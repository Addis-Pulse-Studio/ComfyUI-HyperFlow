#!/usr/bin/env python3
# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""Check a HyperFlow weights file and (optionally) a MiniMax H3 checkpoint before loading them in ComfyUI.

    python tools/inspect_hyperflow.py <hyperflow.safetensors> [<h3 checkpoint.safetensors>]

Reads safetensors headers only (no tensors, no torch): stdlib Python is enough.
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperflow import keys  # noqa: E402
from hyperflow.header import HyperFlowMetadata  # noqa: E402
from hyperflow.schedule import shift_sigmas  # noqa: E402


def read_header(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return header.pop("__metadata__", {}) or {}, header


def inspect_hyperflow(path) -> bool:
    metadata, tensors = read_header(path)
    print(f"HyperFlow file: {path}")
    try:
        meta = HyperFlowMetadata.from_dict(metadata)
    except ValueError as error:
        print(f"  NOT OK: {error}")
        return False
    plan = keys.plan_from_keys(tensors)
    targets = {module: keys.map_module(module) for module in plan}
    ranks = {tensors[pair["A"]]["shape"][0] for pair in plan.values()}
    dtypes = sorted({tensors[k]["dtype"] for k in tensors})
    endpoint = sorted(m for m, t in targets.items() if t.endpoint is not None)
    print(f"  version {meta.version}, gate {meta.gate:.6g}, rank {sorted(ranks)}, alpha {meta.lora_alpha:g}, dtypes {dtypes}")
    print(f"  base model {meta.base_model} @ {meta.base_model_revision or '?'}, tasks {list(meta.tasks)}")
    print(f"  {len(plan)} modules ({len(plan) - len(endpoint)} native targets + {len(endpoint)} endpoint embedder)")
    if meta.sigmas:
        shifted = shift_sigmas(meta.sigmas, meta.video_shift or 12.0)
        print(f"  raw sigmas   {[round(s, 6) for s in meta.sigmas]}")
        print(f"  video sigmas {[round(s, 5) for s in shifted]} (shift {meta.video_shift})")
    else:
        print("  no hyperflow_sigmas in the header: the built-in 8-step grid will be used")
    return True


def inspect_checkpoint(path) -> bool:
    _, tensors = read_header(path)
    names = set(tensors)
    prefix = "model.diffusion_model." if any(k.startswith("model.diffusion_model.") for k in names) else ""
    print(f"H3 checkpoint: {path}")
    if f"{prefix}video_patch_proj.weight" not in names:
        print("  NOT OK: not a ComfyUI-format MiniMax H3 diffusion model")
        return False
    if f"{prefix}adaln_t_table" in names or f"{prefix}time_embedder.proj_in.weight" not in names:
        print("  NOT OK: curve-form checkpoint (adaln_t_table, no time_embedder). Use a non-pruned H3 file.")
        return False
    layers = {int(m.group(1)) for k in names if (m := re.match(rf"{re.escape(prefix)}blocks\.(\d+)\.", k))}
    te = tensors[f"{prefix}time_embedder.proj_in.weight"]
    print(f"  OK: {len(layers)} blocks, time_embedder {te['dtype']} {te['shape']}")
    return True


def main(argv) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 2
    ok = inspect_hyperflow(argv[1])
    if len(argv) == 3:
        ok = inspect_checkpoint(argv[2]) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
