# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""ComfyUI nodes for HyperFlow on MiniMax H3."""

from __future__ import annotations

import json
import logging
import os

import torch

import folder_paths

from .dialogue_nodes import NODE_CLASS_MAPPINGS as DIALOGUE_NODES
from .dialogue_nodes import NODE_DISPLAY_NAME_MAPPINGS as DIALOGUE_NAMES
from .hyperflow.dialogue import (FPS, MAX_TRAINED_FRAMES, MIN_TRAINED_FRAMES, align_frames,
                                 frames_for_seconds)
from .hyperflow.dialogue_audio import seconds_of
from .hyperflow.patch import APPLY_MODES, OPTIONS_KEY, apply_hyperflow
from .hyperflow.schedule import shift_sigmas
from .hyperflow.scheduler import register as register_scheduler

logger = logging.getLogger("HyperFlow")

#: Where pre-converted files (drbaph/Hyperflow-Comfyui) are documented to go; the loader lists it next to loras.
WEIGHTS_FOLDER = "hyperflow"
if WEIGHTS_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[WEIGHTS_FOLDER] = (
        [os.path.join(folder_paths.models_dir, WEIGHTS_FOLDER)], folder_paths.supported_pt_extensions)


#: The one thing users have to understand about this node, so both loaders say it the same way.
APPLY_MODE_TOOLTIP = (
    "How the LoRA goes on. auto (default): merge on a bf16 checkpoint, bypass on a quantized one. patch: merge it "
    "into the weights -- exact and free per step on bf16, but on an int8 checkpoint ComfyUI dequantizes, merges "
    "and then requantizes all 314 weights, which takes about 5 minutes on the first load and quantizes the "
    "adapter a second time. bypass: leave the weights bit-identical to the checkpoint and add up(down(x)) in each "
    "forward -- loads at once, keeps about 3.6 GB of LoRA on the GPU, costs a little time per step. The two are "
    "numerically identical on bf16; on a quantized checkpoint only bypass avoids the second quantization."
)


def _weights_files():
    return sorted(set(folder_paths.get_filename_list("loras")) | set(folder_paths.get_filename_list(WEIGHTS_FOLDER)))


def _weights_path(name):
    return folder_paths.get_full_path("loras", name) or folder_paths.get_full_path_or_raise(WEIGHTS_FOLDER, name)


class HyperFlowLoRALoader:
    DESCRIPTION = (
        "Loads Video Rebirth's HyperFlow 8-step adapter onto a MiniMax H3 model: the LoRA (converted from its "
        "diffusers names), the two-time (t, r) embedder and a per-step endpoint hook. Needs an H3 checkpoint that "
        "still has time_embedder weights (not the *_pruned_* curve-form files). Sample with 'HyperFlow Sigmas', "
        "the euler sampler and cfg 1.0. apply_mode 'auto' bypasses on a quantized checkpoint, where merging the "
        "LoRA would requantize it away, and merges on bf16, where merging is exact and free."
    )
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (_weights_files(), {
                    "tooltip": "The HyperFlow weights file from models/loras or models/hyperflow: Video Rebirth's "
                               "minimax_h3_hyperflow_8step_v1.0.safetensors, or the pre-converted "
                               "custom_node_hyperflow_8step_v1.0_comfyui.safetensors (not the _pruned one)."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Scales both LoRAs (base and endpoint embedder). Only 1.0 is what HyperFlow was trained at."}),
            },
            "optional": {
                "apply_mode": (list(APPLY_MODES), {"default": "auto", "tooltip": APPLY_MODE_TOOLTIP}),
            },
        }

    def load(self, model, lora_name, strength, apply_mode="auto"):
        path = _weights_path(lora_name)
        if strength != 1.0:
            logger.warning("HyperFlow strength %.3g: only 1.0 is validated.", strength)
        return (apply_hyperflow(model, path, strength, apply_mode),)


class HyperFlowLoRALoaderAdvanced(HyperFlowLoRALoader):
    DESCRIPTION = (
        "'HyperFlow LoRA Loader' with the two constants the weights file fixes opened up for ablation: the "
        "two-time blend gate and the sigma grid. The defaults (-1 and empty) use the file's own header, so they "
        "reproduce the released model exactly. Anything else is off-recipe and unvalidated."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        spec["optional"] = dict(spec.get("optional", {}))
        spec["optional"]["gate"] = ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip":
            "Weight of the endpoint embedding in emb_t + gate * (emb_r - emb_t). -1 = the file's header (1.0: "
            "0.25). 0 collapses HyperFlow to single-time conditioning."})
        spec["optional"]["sigmas"] = ("STRING", {"default": "", "multiline": False, "tooltip":
            "The raw (unshifted) sigma grid as a JSON list, e.g. [1.0, 0.93, ..., 0.0]: strictly decreasing, "
            "starting at or below 1.0 and ending at exactly 0.0. Empty = the file's header. 'HyperFlow Sigmas' "
            "and the per-step guard both follow whatever is set here."})
        return spec

    def load(self, model, lora_name, strength, apply_mode="auto", gate=-1.0, sigmas=""):
        path = _weights_path(lora_name)
        if strength != 1.0:
            logger.warning("HyperFlow strength %.3g: only 1.0 is validated.", strength)
        grid = json.loads(sigmas) if sigmas.strip() else None
        if grid is not None and not isinstance(grid, list):
            raise ValueError(f"sigmas must be a JSON list of floats, got {sigmas!r}.")
        gate = None if gate < 0 else float(gate)
        if gate is not None or grid is not None:
            logger.warning("HyperFlow: overriding the weights file's %s; quality is only validated at the file's "
                           "own values.", " and ".join(n for n, v in (("gate", gate), ("sigma grid", grid))
                                                       if v is not None))
        return (apply_hyperflow(model, path, strength, apply_mode, gate=gate, sigmas=grid),)


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


class HyperFlowRetimeAudio:
    DESCRIPTION = (
        "Plays an audio track faster or slower by relabelling its sample rate (no resampling, nothing lost). Use it in "
        "front of a lip-sync node that assumes another frame rate than the video: LatentSync writes its frames at "
        "25 fps, H3 renders at 24 fps, so feeding it the dialogue retimed 24 -> 25 keeps every mouth shape on the "
        "frame that owns it. Play the corrected frames back at 24 fps under the original audio."
    )
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "retime"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                        "tooltip": "The frame rate the frames were rendered at (H3: 24)."}),
                "assumed_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                          "tooltip": "The frame rate the downstream node assumes (LatentSync: 25)."}),
            }
        }

    def retime(self, audio, video_fps, assumed_fps):
        factor = float(assumed_fps) / float(video_fps)
        return ({**audio, "sample_rate": int(round(int(audio["sample_rate"]) * factor))},)


class HyperFlowAudioLength:
    DESCRIPTION = (
        "Reads how long an audio track is and turns it into the scene length, so the video always lasts as long as "
        "the dialogue and there is no duration widget to keep in sync. 'seconds' feeds a duration input (MiniMax H3 "
        "Audio Window's scene_duration_seconds); 'length' is the same duration in frames, snapped up to H3's 17k+5 "
        "grid, for a conditioning node's length. A 13 s take gives 328 frames, a 15 s one 362 - change the file and "
        "everything downstream follows. The bounds are in frames, not seconds, because only whole grid steps exist."
    )
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("seconds", "length", "aligned_seconds", "report")
    FUNCTION = "measure"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The dialogue track the video has to be as long as."}),
                "pad_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":
                    "Extra tail after the audio ends, so the last word is not cut off. 0 suits a clean take."}),
                "min_length": ("INT", {"default": MIN_TRAINED_FRAMES, "min": 5, "max": 100000, "tooltip":
                    "Shortest scene, in frames. H3's shortest trained scene is 124 (5.167 s); a shorter take is "
                    "padded with silence up to here rather than rendered below it."}),
                "max_length": ("INT", {"default": MAX_TRAINED_FRAMES, "min": 0, "max": 100000, "tooltip":
                    "Longest scene, in frames. H3's longest trained scene is 362 (15.083 s); a longer take is cut "
                    "here and the cut is reported. 0 = no cap."}),
                "fps": ("FLOAT", {"default": float(FPS), "min": 1.0, "max": 240.0, "step": 0.001, "tooltip":
                    "The render frame rate (H3: 24)."}),
            }
        }

    def measure(self, audio, pad_seconds, min_length, max_length, fps):
        fps = float(fps)
        audio_seconds = seconds_of(audio)
        seconds = audio_seconds + max(0.0, float(pad_seconds))
        # the bounds are already on the grid, so clamping in seconds cannot land between two steps
        floor, ceiling = align_frames(int(min_length)), int(max_length)
        seconds = max(seconds, floor / fps)
        capped = ceiling > 0 and seconds > ceiling / fps
        if capped:
            seconds = ceiling / fps
        length = frames_for_seconds(seconds, fps)
        aligned = length / fps
        warnings = []
        if capped:
            warnings.append(f"The audio is {audio_seconds:.3f}s but max_length is {ceiling} frames "
                            f"({ceiling / fps:.3f}s): the scene is cut there and the rest of the track will not be "
                            "lip-synced.")
        if audio_seconds + float(pad_seconds) < floor / fps:
            warnings.append(f"The audio is only {audio_seconds:.3f}s; the scene is held at min_length {floor} frames "
                            f"({floor / fps:.3f}s) and the rest is silence.")
        if length > MAX_TRAINED_FRAMES:
            warnings.append(f"{length} frames is past H3's trained maximum of {MAX_TRAINED_FRAMES} "
                            f"({MAX_TRAINED_FRAMES / fps:.3f}s); quality past it is not validated.")
        for warning in warnings:
            logger.warning("HyperFlow audio length: %s", warning)
        # grid rounding is how H3 works, not a problem: report it, do not warn about it
        notes = warnings + [f"The 17k+5 grid rounds {seconds:.3f}s up to {length} frames ({aligned:.3f}s): "
                            f"{aligned - seconds:.3f}s of silence after the scene ends."]
        report = json.dumps({"audio_seconds": round(audio_seconds, 6), "scene_seconds": round(seconds, 6),
                             "length": length, "aligned_seconds": round(aligned, 6), "fps": fps,
                             "sample_rate": int(audio["sample_rate"]), "notes": notes}, indent=2)
        return (seconds, length, aligned, report)


NODE_CLASS_MAPPINGS = {
    "HyperFlowLoRALoader": HyperFlowLoRALoader,
    "HyperFlowLoRALoaderAdvanced": HyperFlowLoRALoaderAdvanced,
    "HyperFlowSigmas": HyperFlowSigmas,
    "HyperFlowRetimeAudio": HyperFlowRetimeAudio,
    "HyperFlowAudioLength": HyperFlowAudioLength,
    **DIALOGUE_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HyperFlowLoRALoader": "HyperFlow LoRA Loader (MiniMax H3)",
    "HyperFlowLoRALoaderAdvanced": "HyperFlow LoRA Loader (Advanced)",
    "HyperFlowSigmas": "HyperFlow Sigmas (8-step)",
    "HyperFlowRetimeAudio": "Retime Audio for Lip-Sync (fps)",
    "HyperFlowAudioLength": "HyperFlow Audio Length (audio \u2192 duration)",
    **DIALOGUE_NAMES,
}

if register_scheduler():
    logger.info("HyperFlow: registered the 'hyperflow' scheduler (8 steps) for scheduler-name samplers.")
