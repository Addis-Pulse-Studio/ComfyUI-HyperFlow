# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""Multi-character dialogue nodes for MiniMax H3: bind each character to its own dialogue line, place the lines on
exact windows, lock the drive audio into the H3 latent, mux the exact final dialogue, and (when H3's own audio
conditioning is not enough) correct one character's lips at a time.

    HyperFlowDialogueLine ×N ─→ HyperFlowDialogueTrack ─┬ drive_audio ─→ HyperFlowAudioLock ─→ sampler
                                                        ├ final_audio ─→ HyperFlowAudioLock ─→ mux_audio ─→ CreateVideo
                                                        ├ length / prompt ─→ MiniMax H3 Reference to Video
                                                        └ plan ─→ HyperFlowSpeakerSegment → LatentSync → HyperFlowSpeakerPaste
"""

from __future__ import annotations

import json
import logging

from .hyperflow.dialogue import FPS, OVERLAP_MODES, DialogueLine, plan_dialogue
from .hyperflow.dialogue_audio import (
    LOCK_MODES,
    av_parts,
    cut_audio,
    cut_segment,
    fit_duration,
    lock_audio_latent,
    paste_segment,
    place_lines,
    seconds_of,
)

logger = logging.getLogger("HyperFlow")

CATEGORY = "HyperFlow/Dialogue"
DIALOGUE = "HF_DIALOGUE"
PLAN = "HF_DIALOGUE_PLAN"
SEGMENT = "HF_SPEAKER_SEGMENT"
SAMPLE_RATES = ["auto", "48000", "44100", "32000"]


def video_seconds(latent: dict) -> float:
    """Duration of an H3 AV latent: latent T = 2 is 5 frames, every 5 more latent frames are 17 more pixel frames."""
    video, audio = av_parts(latent)
    t = int(video.shape[2])
    if t >= 2 and (t - 2) % 5 == 0:
        return (5 + (t - 2) // 5 * 17) / FPS
    return int(audio.shape[-1]) / 40.0


class HyperFlowDialogueLine:
    DESCRIPTION = (
        "One line of dialogue, bound to its speaker: whose face (<Picture N>) and whose voice (<Audio N>) it is, "
        "where it starts on the scene clock and how long it lasts. drive_audio is what moves the mouth; final_audio, "
        "if connected, is the clean take that goes into the MP4 at exactly the same seconds. Chain one node per line."
    )
    CATEGORY = CATEGORY
    RETURN_TYPES = (DIALOGUE,)
    RETURN_NAMES = ("dialogue",)
    FUNCTION = "add"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "drive_audio": ("AUDIO", {"tooltip": "The recording that drives this character's mouth."}),
                "speaker": ("STRING", {"default": "Speaker 1", "tooltip":
                    "The character's name. Lines with the same name are one character, with one id (S1, S2, ...)."}),
                "picture": ("INT", {"default": 1, "min": 0, "max": 9, "tooltip":
                    "This character's <Picture N> in 'MiniMax H3 Reference to Video' (ref_image_{N-1}). 0 = none."}),
                "voice": ("INT", {"default": 1, "min": 0, "max": 9, "tooltip":
                    "This character's voice reference <Audio N> (ref_audio_{N-1}). 0 = none."}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 3600.0, "step": 0.01, "tooltip":
                    "Where the line starts in the scene. -1 = right after the previous line in the chain."}),
                "duration_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3600.0, "step": 0.01, "tooltip":
                    "How long the line lasts. -1 = the rest of the clip. A clip shorter than this is followed by "
                    "silence; a longer one is cut."}),
                "source_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01,
                                                   "tooltip": "Skip this much of the clip before the line starts."}),
            },
            "optional": {
                "dialogue": (DIALOGUE, {"tooltip": "The lines before this one."}),
                "final_audio": ("AUDIO", {"tooltip":
                    "The clean take for the final MP4 (e.g. an unprocessed stem). Must line up with drive_audio "
                    "sample for sample. Defaults to drive_audio."}),
            },
        }

    def add(self, drive_audio, speaker, picture, voice, start_seconds, duration_seconds, source_start_seconds,
            dialogue=None, final_audio=None):
        line = DialogueLine(speaker=speaker, start=float(start_seconds), duration=float(duration_seconds),
                            source_start=float(source_start_seconds), clip_seconds=seconds_of(drive_audio),
                            picture=int(picture), voice=int(voice))
        return (list(dialogue or []) + [{"line": line, "drive": drive_audio, "final": final_audio}],)


class HyperFlowDialogueTrack:
    DESCRIPTION = (
        "Places every dialogue line on its window and returns two tracks of exactly the scene's length (the H3 frame "
        "count / 24): drive_audio for the audio lock, final_audio for the MP4. Also returns that frame count, the "
        "prompt with a timing schedule appended, and the plan the per-character lip-sync nodes read."
    )
    CATEGORY = CATEGORY
    RETURN_TYPES = ("AUDIO", "AUDIO", "INT", "STRING", PLAN, "STRING")
    RETURN_NAMES = ("drive_audio", "final_audio", "length", "prompt", "plan", "report")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue": (DIALOGUE,),
                "scene_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01, "tooltip":
                    "Scene length. 0 = end with the last line. Snapped up to H3's 17k+5 frame grid (15 s -> 362)."}),
                "overlap": (list(OVERLAP_MODES), {"default": "error", "tooltip":
                    "'error' refuses lines that overlap in time; 'mix' allows it and sums them."}),
                "sample_rate": (SAMPLE_RATES, {"default": "auto", "tooltip": "'auto' = the first line's rate."}),
                "prompt_schedule": (["append", "none"], {"default": "append", "tooltip":
                    "Append the dialogue's timing schedule (who speaks when, with their picture and voice tags) to "
                    "the prompt."}),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True, "multiline": True}),
            },
        }

    def build(self, dialogue, scene_seconds, overlap, sample_rate, prompt_schedule, prompt=None):
        if not dialogue:
            raise ValueError("Connect at least one 'HyperFlow Dialogue Line'.")
        plan = plan_dialogue([entry["line"] for entry in dialogue], scene_seconds, overlap)
        rate = 0 if sample_rate == "auto" else int(sample_rate)
        drive, notes = place_lines(plan, {i: e["drive"] for i, e in enumerate(dialogue)}, rate)
        if any(e["final"] is not None for e in dialogue):
            final, _ = place_lines(plan, {i: (e["final"] if e["final"] is not None else e["drive"])
                                          for i, e in enumerate(dialogue)}, int(drive["sample_rate"]))
        else:
            final = drive
        text = prompt or ""
        if prompt_schedule == "append":
            text = (text.rstrip() + "\n\n" if text.strip() else "") + plan.prompt_schedule()
        report = plan.to_dict()
        report["notes"] = notes
        for note in notes:
            logger.warning("HyperFlow dialogue: %s", note)
        return (drive, final, plan.frame_count, text, {"plan": plan, "drive": drive, "final": final},
                json.dumps(report, indent=2))


class HyperFlowAudioLock:
    DESCRIPTION = (
        "Puts the drive audio into the H3 latent's audio stream. 'lock_source' keeps it exactly (audio mask 0) so it "
        "drives the mouths and H3 does not regenerate it; 'remix_source' re-noises it to remix_strength; 'native' "
        "leaves the latent alone. mux_audio is final_audio (else drive_audio) cut to the video's exact length: mux it "
        "instead of the decoded audio to keep your exact dialogue."
    )
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT", "AUDIO", "STRING")
    RETURN_NAMES = ("latent", "mux_audio", "report")
    FUNCTION = "lock"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "From 'MiniMax H3 Reference to Video' / 'Image to Video'."}),
                "audio_vae": ("VAE", {"tooltip": "The MiniMax H3 audio VAE."}),
                "drive_audio": ("AUDIO",),
                "mode": (list(LOCK_MODES), {"default": "lock_source"}),
                "remix_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                    "remix_source only. HyperFlow was not trained on partially masked rows; lock_source is the "
                    "validated mode."}),
            },
            "optional": {
                "final_audio": ("AUDIO", {"tooltip": "The track for the MP4. Defaults to drive_audio."}),
            },
        }

    def lock(self, latent, audio_vae, drive_audio, mode, remix_strength, final_audio=None):
        out, info = lock_audio_latent(latent, audio_vae, drive_audio, mode, remix_strength)
        seconds = video_seconds(latent)
        info["video_seconds"] = round(seconds, 6)
        info["drive_seconds"] = round(seconds_of(drive_audio), 6)
        if abs(info["drive_seconds"] - seconds) > 1.0 / FPS:
            logger.warning("HyperFlow audio lock: the drive audio is %.3fs, the video %.3fs.", info["drive_seconds"],
                           seconds)
        mux = fit_duration(final_audio if final_audio is not None else drive_audio, seconds)
        return (out, mux, json.dumps(info, indent=2))


def _segment_box(region, box_x, box_y, box_w, box_h):
    return None if region == "full_frame" else (box_x, box_y, box_w, box_h)


class HyperFlowSpeakerSegment:
    DESCRIPTION = (
        "Cuts out one character's frames (only the seconds they speak, plus handles) and, optionally, only their face "
        "region, with that character's audio for exactly those frames. Feed both to a lip-sync node (LatentSync), then "
        "'HyperFlow Speaker Paste' puts the corrected frames back. One pair per character; chain them."
    )
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "AUDIO", SEGMENT, "STRING")
    RETURN_NAMES = ("images", "audio", "segment", "report")
    FUNCTION = "cut"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The decoded scene, all frames."}),
                "plan": (PLAN, {"tooltip": "From 'HyperFlow Dialogue Track'."}),
                "speaker": ("STRING", {"default": "Speaker 1", "tooltip": "A speaker name or id (S1, S2, ...)."}),
                "region": (["full_frame", "box"], {"default": "full_frame", "tooltip":
                    "'box' crops to the character's face region, for shots where several faces are visible (the "
                    "lip-sync model then only sees this one)."}),
                "box_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "box_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "box_w": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "box_h": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "handle_seconds": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip":
                    "Extra context before and after each line, so the mouth closes naturally."}),
                "audio_source": (["final", "drive"], {"default": "final"}),
                "lipsync_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.001, "tooltip":
                    "The frame rate the lip-sync node assumes. LatentSync writes 25 fps, so its audio is relabelled "
                    "24 -> 25 (no resampling) to keep each mouth shape on its frame. Set 24 for a 24 fps corrector."}),
            },
        }

    def cut(self, images, plan, speaker, region, box_x, box_y, box_w, box_h, handle_seconds, audio_source,
            lipsync_fps):
        dialogue = plan["plan"]
        notes = []
        if images.shape[0] != dialogue.frame_count:
            notes.append(f"{images.shape[0]} frames for a {dialogue.frame_count}-frame plan.")
        spans = dialogue.speaker_frames(speaker, round(float(handle_seconds) * dialogue.fps))
        frames, segment = cut_segment(images, spans, _segment_box(region, box_x, box_y, box_w, box_h))
        audio = cut_audio(plan[audio_source], segment["spans"], dialogue.fps)
        factor = float(lipsync_fps) / dialogue.fps
        audio = {**audio, "sample_rate": int(round(int(audio["sample_rate"]) * factor))}
        spk = dialogue.speaker(speaker)
        segment["speaker"] = f"{spk.speaker_id} {spk.name}"
        report = {"speaker": segment["speaker"], "spans": segment["spans"], "frames": len(segment["indices"]),
                  "box": segment["box"], "lipsync_fps": lipsync_fps, "notes": notes}
        return (frames, audio, segment, json.dumps(report, indent=2))


class HyperFlowSpeakerPaste:
    DESCRIPTION = (
        "Writes one character's corrected frames back into the scene at exactly the frames and region "
        "'HyperFlow Speaker Segment' cut them from, with a feathered edge. Everyone else's frames are untouched."
    )
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "paste"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The scene the segment was cut from (or the previous paste)."}),
                "corrected_images": ("IMAGE", {"tooltip": "The lip-sync node's output for this segment."}),
                "segment": (SEGMENT,),
                "feather_px": ("INT", {"default": 12, "min": 0, "max": 256, "tooltip":
                    "Blend width at the region's edges (edges on the frame border are not blended)."}),
            },
        }

    def paste(self, images, corrected_images, segment, feather_px):
        out, notes = paste_segment(images, corrected_images, segment, feather_px)
        for note in notes:
            logger.warning("HyperFlow speaker paste (%s): %s", segment.get("speaker", "?"), note)
        return (out, json.dumps({"speaker": segment.get("speaker"), "frames": len(segment["indices"]),
                                 "notes": notes}, indent=2))


NODE_CLASS_MAPPINGS = {
    "HyperFlowDialogueLine": HyperFlowDialogueLine,
    "HyperFlowDialogueTrack": HyperFlowDialogueTrack,
    "HyperFlowAudioLock": HyperFlowAudioLock,
    "HyperFlowSpeakerSegment": HyperFlowSpeakerSegment,
    "HyperFlowSpeakerPaste": HyperFlowSpeakerPaste,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HyperFlowDialogueLine": "HyperFlow Dialogue Line (speaker ↔ audio)",
    "HyperFlowDialogueTrack": "HyperFlow Dialogue Track (timed windows)",
    "HyperFlowAudioLock": "HyperFlow Audio Lock (drive → H3 latent)",
    "HyperFlowSpeakerSegment": "HyperFlow Speaker Segment (lip-sync cut)",
    "HyperFlowSpeakerPaste": "HyperFlow Speaker Paste (lip-sync merge)",
}
