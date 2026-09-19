# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""ComfyUI nodes for HyperFlow on MiniMax H3."""

from __future__ import annotations

import logging

import torch

import folder_paths

from .hyperflow.patch import OPTIONS_KEY, apply_hyperflow
from .hyperflow.schedule import shift_sigmas

logger = logging.getLogger("HyperFlow")


class HyperFlowLoRALoader:
    DESCRIPTION = (
        "Loads Video Rebirth's HyperFlow 8-step adapter onto a MiniMax H3 model: the LoRA (converted from its "
        "diffusers names), the two-time (t, r) embedder and a per-step endpoint hook. Needs an H3 checkpoint that "
        "still has time_embedder weights (not the *_pruned_* curve-form files). Sample with 'HyperFlow Sigmas', "
        "the euler sampler and cfg 1.0."
    )
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "The HyperFlow weights file, e.g. minimax_h3_hyperflow_8step_v1.0.safetensors."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Scales both LoRAs (base and endpoint embedder). Only 1.0 is what HyperFlow was trained at."}),
            }
        }

    def load(self, model, lora_name, strength):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if strength != 1.0:
            logger.warning("HyperFlow strength %.3g: only 1.0 is validated.", strength)
        return (apply_hyperflow(model, path, strength),)


class HyperFlowSigmas:
    DESCRIPTION = (
        "The fixed 8-step sigma grid stored in the HyperFlow weights file, shifted with the model's video shift "
        "(12.0). There is no step count: HyperFlow was distilled onto this grid only."
    )
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL", {"tooltip": "The model from 'HyperFlow LoRA Loader'."})}}

    def get_sigmas(self, model):
        options = model.model_options.get(OPTIONS_KEY)
        if options is None:
            raise ValueError("This model has no HyperFlow loaded; connect the output of 'HyperFlow LoRA Loader'.")
        sampling = model.get_model_object("model_sampling")
        shift = float(sampling.shift)
        audio_shift = getattr(sampling, "audio_shift", None)
        trained = (options["video_shift"], options["audio_shift"])
        if (shift, audio_shift) != trained:
            logger.warning(
                "HyperFlow was trained with shifts video %.3g / audio %.3g; this model samples with %.3g / %s. "
                "Quality is only validated at the trained shifts.", trained[0], trained[1], shift, audio_shift,
            )
        return (torch.tensor(shift_sigmas(options["sigmas"], shift), dtype=torch.float32),)


NODE_CLASS_MAPPINGS = {
    "HyperFlowLoRALoader": HyperFlowLoRALoader,
    "HyperFlowSigmas": HyperFlowSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HyperFlowLoRALoader": "HyperFlow LoRA Loader (MiniMax H3)",
    "HyperFlowSigmas": "HyperFlow Sigmas (8-step)",
}
