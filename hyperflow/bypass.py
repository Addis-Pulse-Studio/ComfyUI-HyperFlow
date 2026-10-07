# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""HyperFlow's LoRA as forward hooks instead of weight patches.

Merging a LoRA into a quantized checkpoint quantizes it a second time. ComfyUI's
``ModelPatcher.patch_weight_to_device`` dequantizes the weight, adds the patch, then hands the result to the op's
``set_weight``, which for a quantized Linear is ``requantize_from_float(w, scale="recalculate", ...)``. On an
``*_int8_convrot`` checkpoint that re-runs comfy-kitchen's ``quantize(convrot=True, per_channel=True)``, so it is
a properly calibrated second pass rather than a crude round-trip -- but it is still a second pass, over all 314
weights, and it costs about five minutes of first load. Bypass leaves the checkpoint's weights bit-identical and
adds ``up(down(x))`` in the activation's own dtype, which is what HyperFlow was distilled as, so it cannot
introduce that error at all. How much the error is worth in practice is not measured yet; see
``docs/VERIFICATION.md``.

The hook machinery is ComfyUI's (``comfy.weight_adapter.BypassInjectionManager``); what this module adds is an
adapter that is *exactly* the weight patch it replaces:

* **fused q|k|v.** HyperFlow patches three row slices of ``attn.qkv_proj``. One adapter holds all three and
  concatenates their outputs along the output features -- no block-diagonal zero padding to store or multiply.
* **folded input activations.** ComfyUI calls ``mlp.fc2(mlp.fc1(x), input_act="swiglu")``: the SwiGLU rides inside
  fc2's own (possibly INT8) kernel, so a hook on fc2 sees the *pre*-activation tensor, twice as wide as fc2's
  input. The base computes ``W @ act(x)``, so the merged weight would add ``s * B @ (A @ act(x))`` -- the adapter
  applies ComfyUI's own activation first and is therefore exact, instead of crashing on the width or (before
  ComfyUI PR #16816, which routed the fused path around ``fc2.forward`` entirely) being silently dropped.
* **folded residuals.** ``linear(..., residual=r, residual_scale=s)`` returns ``r + s * linear(act(x))``, so the
  delta has to carry the same ``s``. No HyperFlow target is called that way today; #16816 shows the direction.

Anything else ComfyUI starts folding into a Linear's forward raises, naming ``apply_mode 'patch'``, rather than
quietly computing the wrong delta.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("HyperFlow")

#: What ComfyUI's ``Linear.forward`` accepts after the input, in order (``comfy/ops.py``). Everything here is
#: mirrored into the LoRA branch; anything else is refused.
LINEAR_EXTRAS = ("input_act", "act_weight", "residual", "residual_scale")

_PATCH_INSTEAD = "Load the HyperFlow LoRA with apply_mode 'patch' instead, and please report this."

#: ``ModelPatcher.set_injections`` key the hooks live under, so ComfyUI's own unpatching ejects them.
INJECTION_KEY = "hyperflow_bypass"


def module_of(model, weight_key: str):
    """The module that owns ``<module path>.weight``, or None if this model has no such module."""
    import comfy.utils

    path = weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key
    try:
        return comfy.utils.get_attr(model.model, path)
    except AttributeError:
        return None


def _features(module, weight_key: str, which: str) -> int:
    """``in_features`` / ``out_features`` of a Linear, from the module rather than its stored weight.

    A quantized module's ``weight`` is packed storage -- int4 and int6 formats store ``K * bits / 8`` columns --
    so its shape is not the layer's logical geometry and cannot be compared against a LoRA's.
    """
    value = getattr(module, f"{which}_features", None)
    if value is not None:
        return int(value)
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        raise ValueError(f"HyperFlow: {weight_key} is not a Linear with {which}_features; cannot validate it.")
    if is_quantized(module):
        raise ValueError(f"HyperFlow: {weight_key} is quantized and has no {which}_features to validate against.")
    return int(weight.shape[0 if which == "out" else 1])


def in_features(module, weight_key: str) -> int:
    return _features(module, weight_key, "in")


def out_features(module, weight_key: str) -> int:
    return _features(module, weight_key, "out")


def is_quantized(module) -> bool:
    """Whether this module keeps its weight in a quantized layout, so merging a patch would requantize it."""
    import comfy.ops

    if getattr(module, "quant_format", None) is not None:
        return True
    return isinstance(getattr(module, "weight", None), comfy.ops.QuantizedTensor)


def lora_scale(adapter) -> float:
    """``alpha / rank`` of one ComfyUI LoRA adapter, refusing the variants bypass cannot reproduce."""
    up, down, alpha, mid, dora_scale = adapter.weights[:5]
    if mid is not None or dora_scale is not None:
        raise ValueError("HyperFlow bypass supports plain LoRA patches only (no mid / DoRA).")
    return 1.0 if alpha is None else float(alpha) / down.shape[0]


def linear_extras(weight_key: str, args, kwargs) -> tuple:
    """``(input_act, act_weight, residual, residual_scale)`` of one Linear call, however it was passed."""
    if len(args) > len(LINEAR_EXTRAS):
        raise ValueError(f"HyperFlow bypass: {weight_key} was called with {len(args)} extra positional arguments, "
                         f"more than ComfyUI's Linear.forward takes ({len(LINEAR_EXTRAS)}). {_PATCH_INSTEAD}")
    values = dict(zip(LINEAR_EXTRAS, args))
    for name, value in kwargs.items():
        if name not in LINEAR_EXTRAS:
            raise ValueError(f"HyperFlow bypass: {weight_key} was called with {name!r}, which this adapter does "
                             f"not know how to mirror into the LoRA branch. {_PATCH_INSTEAD}")
        if name in values:
            raise ValueError(f"HyperFlow bypass: {weight_key} got {name!r} both positionally and by keyword.")
        values[name] = value
    return tuple(values.get(name) for name in LINEAR_EXTRAS)


def apply_input_act(weight_key: str, x: torch.Tensor, act, act_weight) -> torch.Tensor:
    """``x`` with the activation ComfyUI folds into this Linear, using ComfyUI's own implementation of it."""
    if act is None:
        return x
    import comfy.ops

    function = comfy.ops.INPUT_ACT_EAGER.get(act)
    if function is None:
        # "rms_norm" lives outside INPUT_ACT_EAGER because it needs the norm module's own weight cast context.
        raise ValueError(f"HyperFlow bypass: {weight_key} folds a {act!r} activation into its forward, which this "
                         f"adapter cannot reproduce exactly. {_PATCH_INSTEAD}")
    return function(x)


class HyperFlowBypass:
    """One bypass adapter per native weight; a fused ``qkv_proj`` carries q, k and v as three row parts.

    ``delta(x) = cat_i[ up_i(down_i(act(x))) * s_i ]`` along the output features, which is what the row-slice
    weight patches it replaces would have added. ComfyUI's ``BypassForwardHook`` owns the ``multiplier`` (the
    loader's ``strength``) and moves ``weights`` to the compute device once, at inject time.
    """

    def __init__(self, parts, weight_key: str = ""):
        if not parts:
            raise ValueError(f"HyperFlow bypass: no LoRA parts for {weight_key}.")
        self.key = weight_key
        self.scales = tuple(scale for _, _, scale in parts)
        # flat (up, down, up, down, ...): the shape ComfyUI's hook knows how to move between devices
        self.weights = tuple(tensor for up, down, _ in parts for tensor in (up, down))
        self.multiplier = 1.0

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i, scale in enumerate(self.scales):
            up = self.weights[2 * i].to(device=x.device, dtype=x.dtype)      # no-ops once the hook has moved them
            down = self.weights[2 * i + 1].to(device=x.device, dtype=x.dtype)
            outs.append(torch.nn.functional.linear(torch.nn.functional.linear(x, down), up) * scale)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        return out * self.multiplier

    def bypass_forward(self, org_forward, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """ComfyUI calls this in place of ``h``/``g`` because it is overridden, so the folded arguments arrive here."""
        # validated before the base runs, so an argument we cannot mirror raises our message, not a TypeError
        act, act_weight, residual, residual_scale = linear_extras(self.key, args, kwargs)
        base = org_forward(x, *args, **kwargs)
        delta = self.delta(apply_input_act(self.key, x, act, act_weight))
        if residual is not None and residual_scale is not None:
            delta = delta * residual_scale  # base is residual + residual_scale * linear(act(x))
        return base + delta.to(base.dtype)

    # Unused while bypass_forward is overridden; kept so the adapter still satisfies ComfyUI's plain protocol.
    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        return self.delta(x)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        return y


def group_patches(patches) -> dict[str, list]:
    """ComfyUI's patch dict -> ``{native weight key: [(row start, rows or None, adapter)]}``."""
    import comfy.weight_adapter

    grouped: dict[str, list] = {}
    for key, adapter in patches.items():
        if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            raise ValueError(f"HyperFlow bypass: {key} is a {type(adapter).__name__}, not a LoRA.")
        if isinstance(key, str):
            grouped.setdefault(key, []).append((0, None, adapter))
            continue
        weight_key, (dim, start, rows) = key[0], key[1]
        if dim != 0:
            raise ValueError(f"HyperFlow bypass: {weight_key} is sliced along dim {dim}; only rows are supported.")
        grouped.setdefault(weight_key, []).append((start, rows, adapter))
    return grouped


def quantized_count(model, patches) -> int:
    """How many of these patch targets the checkpoint stores quantized (so merging would requantize them)."""
    return sum(1 for key in group_patches(patches) if is_quantized(module_of(model, key)))


def install(model, patches, strength: float) -> dict:
    """Hook every HyperFlow LoRA onto ``model`` (a clone). Returns the patches that need the merge path instead.

    A target ComfyUI cannot resolve to a hookable module -- a layout that folds the weight into a parent's fused
    kernel -- is handed back rather than dropped, so the loader can merge exactly those and say so.
    """
    import comfy.weight_adapter

    grouped = group_patches(patches)
    originals: dict[str, list] = {}
    for key in patches:
        originals.setdefault(key if isinstance(key, str) else key[0], []).append(key)

    manager = comfy.weight_adapter.BypassInjectionManager()
    for weight_key, parts in sorted(grouped.items()):
        module = module_of(model, weight_key)
        if module is None:
            continue  # reported as a merge target below, with the rest ComfyUI could not hook
        rows_total = out_features(module, weight_key)
        parts.sort(key=lambda part: part[0])
        covered = 0
        for start, rows, _ in parts:
            if start != covered:
                raise ValueError(f"HyperFlow bypass: the row patches of {weight_key} do not tile it "
                                 f"(gap at {covered}).")
            covered += rows_total if rows is None else rows
        if covered != rows_total:
            raise ValueError(f"HyperFlow bypass: the row patches of {weight_key} cover {covered} of "
                             f"{rows_total} rows.")
        adapter = HyperFlowBypass([(a.weights[0], a.weights[1], lora_scale(a)) for _, _, a in parts], weight_key)
        manager.add_adapter(weight_key, adapter, strength)

    injections = manager.create_injections(model.model)
    hooked = {id(hook.adapter) for hook in manager.hooks}
    for hook in manager.hooks:
        if getattr(hook.adapter, "is_conv", False):
            raise ValueError(f"HyperFlow bypass: {hook.adapter.key} is a convolution, not a Linear; its output "
                             f"features are not the last dimension. {_PATCH_INSTEAD}")
    model.set_injections(INJECTION_KEY, injections)

    unhooked = {adapter.key for adapter, _ in manager.adapters.values() if id(adapter) not in hooked}
    unhooked |= {key for key in grouped if module_of(model, key) is None}
    return {key: patches[key] for weight_key in sorted(unhooked) for key in originals[weight_key]}
