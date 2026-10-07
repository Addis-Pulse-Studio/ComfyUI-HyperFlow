#!/usr/bin/env python3
# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""Regenerate example_workflows/*.json (ComfyUI UI format). Stdlib only.

    python tools/make_example_workflows.py

Model and media names are the ones the examples expect under ComfyUI/models and ComfyUI/input; change the widgets
after loading if yours differ.
"""

from __future__ import annotations

import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_workflows")

FL2VA = "minimax\\higher\\minimax_h3_fl2va_int8_convrot.safetensors"
REF2VA = "minimax\\higher\\minimax_h3_ref2va_int8_convrot.safetensors"
HYPERFLOW = "minimaxh3\\minimax_h3_hyperflow_8step_v1.0.safetensors"
CLIP = "minimax\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax\\minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax\\minimax_h3_audio_vae_fp32.safetensors"
SINGULARITY = "minimax\\Singularity\\Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors"

COMMON_NOTE = """**Checkpoint:** a non-pruned H3 (`*_int8_convrot` or `*_bf16`). The `*_pruned_*` files are curve-form (no
`time_embedder`) and the loader refuses them.

**Weights:** `hf download videorebirth/hyperflow minimax_h3_hyperflow_8step_v1.0.safetensors
--local-dir models/loras/minimaxh3` (MiniMax H3 Community License; read it first).

**Sampling is fixed:** `HyperFlow Sigmas` gives the 8-step grid stored in the file; sampler `euler`;
`BasicGuider` (cfg 1.0). No steps, no scheduler, no CFG. Another schedule or a multi-stage sampler raises.

Don't stack HyperFlow with FastH3 / Turbo / TaoMate step LoRAs."""


class Graph:
    def __init__(self):
        self.nodes, self.links, self.groups, self.by_id = [], [], [], {}
        # monotonic, never len(links) + 1: the dialogue graph deletes a link, which would otherwise reissue its id
        self.next_link = 1

    def add(self, i, kind, pos, widgets=None, inputs=(), outputs=(), size=(320, 100), title=None):
        node = {"id": i, "type": kind, "pos": list(pos), "size": list(size), "flags": {}, "order": len(self.nodes),
                "mode": 0, "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
                "outputs": [{"name": n, "type": t, "links": []} for n, t in outputs],
                "properties": {"Node name for S&R": kind}, "widgets_values": widgets if widgets is not None else []}
        if title:
            node["title"] = title
        self.nodes.append(node)
        self.by_id[i] = node

    def link(self, src, slot, dst, name):
        lid, self.next_link = self.next_link, self.next_link + 1
        s, d = self.by_id[src], self.by_id[dst]
        kind = s["outputs"][slot]["type"]
        dslot = next(j for j, x in enumerate(d["inputs"]) if x["name"] == name)
        s["outputs"][slot]["links"].append(lid)
        d["inputs"][dslot]["link"] = lid
        self.links.append([lid, src, slot, dst, dslot, kind])

    def group(self, title, bounding, color):
        self.groups.append({"id": len(self.groups) + 1, "title": title, "bounding": list(bounding), "color": color,
                            "font_size": 24, "flags": {}})

    def dump(self, name):
        wf = {"id": f"hyperflow-{name}", "revision": 0, "last_node_id": max(self.by_id),
              "last_link_id": self.next_link - 1,
              "nodes": self.nodes, "links": self.links, "groups": self.groups, "config": {},
              "extra": {"ds": {"scale": 0.75, "offset": [0, 0]}}, "version": 0.4}
        with open(os.path.join(OUT, f"hyperflow_{name}.json"), "w", encoding="utf-8") as handle:
            json.dump(wf, handle, indent=1, ensure_ascii=False)
            handle.write("\n")


def base(g: Graph, checkpoint: str, prefix: str):
    """Loaders, HyperFlow, sampler and decode/save. Conditioning node 7 is added by the caller."""
    g.add(1, "UNETLoader", (40, 60), [checkpoint, "default"], outputs=[("MODEL", "MODEL")], size=(380, 90))
    g.add(2, "HyperFlowLoRALoader", (460, 60), [HYPERFLOW, 1.0, "auto"], [("model", "MODEL")], [("MODEL", "MODEL")],
          size=(380, 90))
    g.add(3, "CLIPLoader", (40, 200), [CLIP, "minimax", "default"], outputs=[("CLIP", "CLIP")], size=(380, 110))
    g.add(4, "VAELoader", (40, 350), [VIDEO_VAE], outputs=[("VAE", "VAE")], size=(380, 60), title="Video VAE")
    g.add(5, "VAELoader", (40, 450), [AUDIO_VAE], outputs=[("VAE", "VAE")], size=(380, 60), title="Audio VAE")
    g.add(8, "HyperFlowSigmas", (900, 60), [], [("model", "MODEL")], [("SIGMAS", "SIGMAS")], size=(300, 50))
    g.add(9, "KSamplerSelect", (900, 160), ["euler"], outputs=[("SAMPLER", "SAMPLER")], size=(300, 60))
    g.add(10, "RandomNoise", (900, 260), [42, "fixed"], outputs=[("NOISE", "NOISE")], size=(300, 80))
    g.add(11, "BasicGuider", (900, 380), [], [("model", "MODEL"), ("conditioning", "CONDITIONING")],
          [("GUIDER", "GUIDER")], size=(300, 50))
    g.add(12, "SamplerCustomAdvanced", (1240, 60), [],
          [("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
           ("latent_image", "LATENT")], [("output", "LATENT"), ("denoised_output", "LATENT")], size=(300, 110))
    g.add(13, "VAEDecode", (1240, 220), [], [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], size=(220, 50))
    g.add(14, "VAEDecodeAudio", (1240, 320), [], [("samples", "LATENT"), ("vae", "VAE")], [("AUDIO", "AUDIO")], size=(220, 50))
    g.add(15, "CreateVideo", (1500, 220), [24, 8, "sRGB"], [("images", "IMAGE"), ("audio", "AUDIO")],
          [("VIDEO", "VIDEO")], size=(260, 100))
    g.add(16, "SaveVideo", (1500, 380), [f"video/hyperflow_{prefix}", "auto", "auto"], [("video", "VIDEO")],
          size=(320, 320))
    g.link(1, 0, 2, "model")
    g.link(2, 0, 8, "model")
    g.link(2, 0, 11, "model")
    g.link(10, 0, 12, "noise")
    g.link(11, 0, 12, "guider")
    g.link(9, 0, 12, "sampler")
    g.link(8, 0, 12, "sigmas")
    g.link(12, 0, 13, "samples")
    g.link(4, 0, 13, "vae")
    g.link(12, 0, 14, "samples")
    g.link(5, 0, 14, "vae")
    g.link(13, 0, 15, "images")
    g.link(14, 0, 15, "audio")
    g.link(15, 0, 16, "video")


def as_input(g: Graph, node: int, *names: str):
    """Mark widgets as widget-converted inputs, so another node can drive them."""
    for slot in g.by_id[node]["inputs"]:
        if slot["name"] in names:
            slot["widget"] = {"name": slot["name"]}


def image_to_video(g: Graph, prompt: str, length: int = 124, length_input: bool = False):
    inputs = [("clip", "CLIP"), ("vae", "VAE"), ("first_frame", "IMAGE"), ("last_frame", "IMAGE")]
    if length_input:
        inputs.append(("length", "INT"))
    g.add(7, "MiniMaxH3ImageToVideo", (460, 200), [prompt, 1344, 768, length], inputs,
          [("positive", "CONDITIONING"), ("LATENT", "LATENT")], size=(400, 300))
    if length_input:
        as_input(g, 7, "length")
    g.link(3, 0, 7, "clip")
    g.link(4, 0, 7, "vae")
    g.link(7, 1, 12, "latent_image")


def audio_length(g: Graph, i: int, pos, audio_node: int, title="Scene length = the audio's length"):
    """'HyperFlow Audio Length': the render lasts exactly as long as the track, with no duration widget to sync."""
    g.add(i, "HyperFlowAudioLength", pos, [0.0, 124, 362, 24.0], [("audio", "AUDIO")],
          [("seconds", "FLOAT"), ("length", "INT"), ("aligned_seconds", "FLOAT"), ("report", "STRING")],
          size=(320, 180), title=title)
    g.link(audio_node, 0, i, "audio")


def t2va():
    g = Graph()
    base(g, FL2VA, "t2va")
    image_to_video(g, "A red fox trots through a snowy pine forest at dawn, breath steaming in the cold air. "
                      "The soft crunch of snow under its paws and distant birdsong.")
    g.link(7, 0, 11, "conditioning")
    g.add(17, "MarkdownNote", (460, 560), ["# HyperFlow 8-step · MiniMax H3 text → video + audio (t2va)\n\n"
                                           "t2va runs on the fl2va checkpoint with no frames connected: H3 writes "
                                           "the soundtrack from the prompt along with the picture.\n\n" + COMMON_NOTE],
          size=(520, 400), title="Read me")
    g.dump("t2va")


def fl2va():
    g = Graph()
    base(g, FL2VA, "fl2va")
    image_to_video(g, "The subject walks forward through the scene and turns toward the camera, gentle ambient wind "
                      "and footsteps.")
    g.add(6, "LoadImage", (40, 560), ["example.png", "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
          size=(320, 320), title="First frame")
    g.add(18, "LoadImage", (40, 920), ["example.png", "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
          size=(320, 320), title="Last frame (optional)")
    g.link(6, 0, 7, "first_frame")
    g.link(18, 0, 7, "last_frame")
    # Optional external soundtrack anchored at frame 0 (voice / music the video must follow).
    g.add(19, "LoadAudio", (460, 560), ["", None, None], [], [("AUDIO", "AUDIO")], size=(320, 140),
          title="External audio (optional)")
    g.add(20, "MiniMaxH3AddGuide", (900, 480), [0],
          [("positive", "CONDITIONING"), ("vae", "VAE"), ("audio_vae", "VAE"), ("latent", "LATENT"),
           ("image", "IMAGE"), ("audio", "AUDIO")], [("positive", "CONDITIONING")], size=(300, 160),
          title="Anchor audio at frame 0 (bypass if unused)")
    g.by_id[20]["mode"] = 4  # bypassed until an audio file is chosen
    g.link(7, 0, 20, "positive")
    g.link(7, 1, 20, "latent")
    g.link(5, 0, 20, "audio_vae")
    g.link(19, 0, 20, "audio")
    g.link(20, 0, 11, "conditioning")
    g.add(17, "MarkdownNote", (460, 740), ["# HyperFlow 8-step · MiniMax H3 first/last frame → video + audio\n\n"
                                           "Disconnect *Last frame* for plain i2v. To drive the scene with your own "
                                           "soundtrack, pick a file in *External audio* and un-bypass (Ctrl+B) the "
                                           "*Anchor audio* node: the track is encoded as conditioning rows pinned at "
                                           "frame 0 and the generated video follows it.\n\n" + COMMON_NOTE],
          size=(520, 420), title="Read me")
    g.dump("fl2va")


def ref2va():
    g = Graph()
    base(g, REF2VA, "ref2va")
    g.add(6, "LoadImage", (40, 560), ["example.png", "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
          size=(320, 320), title="Reference image (character)")
    g.add(19, "LoadAudio", (40, 920), ["", None, None], [], [("AUDIO", "AUDIO")], size=(320, 140),
          title="Reference audio (voice)")
    g.add(7, "MiniMaxH3ReferenceToVideo", (460, 200),
          ["The character from <Picture 1> talks to the camera in the voice from <Audio 1>, lips in sync with the "
           "words, soft studio light.", 1344, 768, 124, "match"],
          [("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"), ("ref_images.ref_image_0", "IMAGE"),
           ("ref_audios.ref_audio_0", "AUDIO"), ("length", "INT")],
          [("positive", "CONDITIONING"), ("LATENT", "LATENT")], size=(400, 320))
    as_input(g, 7, "length")
    audio_length(g, 26, (400, 1100), 19)
    g.link(3, 0, 7, "clip")
    g.link(4, 0, 7, "vae")
    g.link(5, 0, 7, "audio_vae")
    g.link(6, 0, 7, "ref_images.ref_image_0")
    g.link(19, 0, 7, "ref_audios.ref_audio_0")
    g.link(26, 1, 7, "length")
    g.link(7, 0, 11, "conditioning")
    g.link(7, 1, 12, "latent_image")
    g.add(17, "MarkdownNote", (460, 580), ["# HyperFlow 8-step · MiniMax H3 reference image + audio → video + audio\n\n"
                                           "Avatar / character animation on the **ref2va** checkpoint. The same "
                                           "HyperFlow file works: it never touches `adaln_proj`, which carries the "
                                           "reference conditioning. Reference rows stay pinned (r = t).\n\n"
                                           "`<Audio 1>` is the **reference voice**: H3 speaks the prompt in it. "
                                           "*Audio Length* makes the scene last as long as that clip (snapped up to "
                                           "the 17k+5 grid), so a 13 s take renders 328 frames instead of the "
                                           "124-frame default. Disconnect it to set `length` by hand.\n\n"
                                           "To make the character say an **exact recording** rather than a voice "
                                           "like it, use `hyperflow_audio2va.json` (anchored track) or "
                                           "`hyperflow_ref2va_2speaker_dialogue.json` (audio lock).\n\n" + COMMON_NOTE],
          size=(520, 520), title="Read me")
    g.dump("ref2va")


def audio2va():
    """An external recording drives the scene: anchored at frame 0, and it sets the scene's length."""
    g = Graph()
    base(g, FL2VA, "audio2va")
    image_to_video(g, "A woman in a sunlit kitchen speaks directly to camera, lips in sync with every word of the "
                      "supplied recording, warm morning light, quiet room tone. No music, no other voices.",
                   length_input=True)
    g.add(19, "LoadAudio", (40, 560), ["", None, None], [], [("AUDIO", "AUDIO")], size=(320, 140),
          title="Voice / soundtrack the video must follow")
    audio_length(g, 26, (40, 740), 19)
    g.add(18, "LoadImage", (40, 960), ["example.png", "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
          size=(320, 320), title="First frame (optional · Ctrl+B to enable)")
    g.by_id[18]["mode"] = 4
    g.add(20, "MiniMaxH3AddGuide", (900, 480), [0],
          [("positive", "CONDITIONING"), ("vae", "VAE"), ("audio_vae", "VAE"), ("latent", "LATENT"),
           ("image", "IMAGE"), ("audio", "AUDIO")], [("positive", "CONDITIONING")], size=(300, 160),
          title="Anchor the track at frame 0")
    for src, slot, dst, name in ((26, 1, 7, "length"), (7, 0, 20, "positive"), (7, 1, 20, "latent"),
                                 (4, 0, 20, "vae"), (5, 0, 20, "audio_vae"), (18, 0, 20, "image"),
                                 (19, 0, 20, "audio"), (20, 0, 11, "conditioning")):
        g.link(src, slot, dst, name)
    g.add(17, "MarkdownNote", (460, 560), [
        "# HyperFlow 8-step · external audio → synchronized video + audio\n\n"
        "**Pick an audio file and press Run** — everything else is wired.\n\n"
        "- *Add Guide* encodes the track with the audio VAE and anchors it at frame 0, as conditioning rows. "
        "HyperFlow pins those rows (`r = t`) while the generated rows step along the grid, so the picture follows "
        "the real waveform.\n"
        "- *Audio Length* makes the scene exactly as long as the track: `length` is the audio's duration in frames, "
        "snapped up to H3's 17k+5 grid (a 13 s take → 328 frames / 13.667 s). Read `report` for what it did; the "
        "trailing silence is the grid rounding, not an error.\n"
        "- H3 **regenerates** the soundtrack rather than copying your samples, so phase differs even though timing "
        "and loudness match (measured: envelope correlation 0.97, 0 ms lag). Mux your own file over the frames if "
        "you need the original samples, or use the audio lock in "
        "`hyperflow_ref2va_2speaker_dialogue.json`.\n"
        "- Enable *First frame* (Ctrl+B) to pin the opening frame as well — the configuration in "
        "`docs/VERIFICATION.md`.\n\n" + COMMON_NOTE], size=(520, 560), title="Read me")
    g.dump("audio2va")


def dialogue():
    """Two characters, each bound to their own dialogue line, with the lines locked into the audio latent."""
    g = Graph()
    base(g, REF2VA, "ref2va_2speaker_dialogue")
    # H3's decoded audio is not what goes in the film: the audio lock's mux_audio is.
    (decoded,) = [lk for lk in g.links if lk[1] == 14 and lk[3] == 15]
    g.links.remove(decoded)
    g.by_id[14]["outputs"][0]["links"].remove(decoded[0])
    g.by_id[15]["inputs"][1]["link"] = None
    g.by_id[14]["mode"] = 4
    g.by_id[14]["title"] = "H3's decoded audio (unused: the dialogue is muxed exactly)"

    for i, y, who in ((6, 560, "Picture 1 · woman"), (18, 900, "Picture 2 · man")):
        g.add(i, "LoadImage", (40, y), ["example.png", "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
              size=(320, 300), title=who)
    for i, y, who in ((19, 1240, "Audio 1 · woman dialogue"), (20, 1420, "Audio 2 · man dialogue")):
        g.add(i, "LoadAudio", (40, y), ["", None, None], [], [("AUDIO", "AUDIO")], size=(320, 140), title=who)
    line_in = [("drive_audio", "AUDIO"), ("dialogue", "HF_DIALOGUE"), ("final_audio", "AUDIO")]
    g.add(21, "HyperFlowDialogueLine", (400, 1240), ["Woman", 1, 1, 0.0, 7.5, 0.0], line_in,
          [("dialogue", "HF_DIALOGUE")], size=(360, 220), title="Line 1 · Woman = <Picture 1> + <Audio 1>")
    g.add(22, "HyperFlowDialogueLine", (400, 1490), ["Man", 2, 2, -1.0, 7.5, 0.0], line_in,
          [("dialogue", "HF_DIALOGUE")], size=(360, 220), title="Line 2 · Man = <Picture 2> + <Audio 2>, right after")
    g.add(23, "PrimitiveStringMultiline", (400, 560),
          ["<Picture 1> is the woman and <Audio 1> is her voice. <Picture 2> is the man and <Audio 2> is his voice.\n\n"
           "[Shot 1] Only <Picture 1>, the woman, speaks directly to camera in a front-facing studio shot; her lips "
           "match every word. [Shot 2] At 00:07.500, the camera cuts to <Picture 2>, the man, who answers directly "
           "to camera; his lips match every word. No music, no other voices, no subtitles."],
          outputs=[("STRING", "STRING")], size=(360, 300), title="Prompt")
    g.add(24, "HyperFlowDialogueTrack", (800, 1240), [15.0, "error", "auto", "append"],
          [("dialogue", "HF_DIALOGUE"), ("prompt", "STRING")],
          [("drive_audio", "AUDIO"), ("final_audio", "AUDIO"), ("length", "INT"), ("prompt", "STRING"),
           ("plan", "HF_DIALOGUE_PLAN"), ("report", "STRING")], size=(360, 200), title="Dialogue track · timed windows")
    g.add(7, "MiniMaxH3ReferenceToVideo", (460, 200), ["", 1344, 768, 362, "match"],
          [("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"), ("ref_images.ref_image_0", "IMAGE"),
           ("ref_images.ref_image_1", "IMAGE"), ("ref_audios.ref_audio_0", "AUDIO"),
           ("ref_audios.ref_audio_1", "AUDIO"), ("prompt", "STRING"), ("length", "INT")],
          [("positive", "CONDITIONING"), ("LATENT", "LATENT")], size=(400, 320))
    as_input(g, 7, "prompt", "length")
    g.add(25, "HyperFlowAudioLock", (900, 480), ["lock_source", 0.35],
          [("latent", "LATENT"), ("audio_vae", "VAE"), ("drive_audio", "AUDIO"), ("final_audio", "AUDIO")],
          [("latent", "LATENT"), ("mux_audio", "AUDIO"), ("report", "STRING")], size=(300, 130),
          title="Audio lock · lock_source")
    for src, slot, dst, name in (
            (19, 0, 21, "drive_audio"), (20, 0, 22, "drive_audio"), (21, 0, 22, "dialogue"),
            (22, 0, 24, "dialogue"), (23, 0, 24, "prompt"),
            (3, 0, 7, "clip"), (4, 0, 7, "vae"), (5, 0, 7, "audio_vae"), (6, 0, 7, "ref_images.ref_image_0"),
            (18, 0, 7, "ref_images.ref_image_1"), (19, 0, 7, "ref_audios.ref_audio_0"),
            (20, 0, 7, "ref_audios.ref_audio_1"), (24, 3, 7, "prompt"), (24, 2, 7, "length"),
            (7, 0, 11, "conditioning"), (7, 1, 25, "latent"), (5, 0, 25, "audio_vae"), (24, 0, 25, "drive_audio"),
            (24, 1, 25, "final_audio"), (25, 0, 12, "latent_image"), (25, 1, 15, "audio")):
        g.link(src, slot, dst, name)

    # per-character lip-sync correction, bypassed until native sync falls short
    seg_in = [("images", "IMAGE"), ("plan", "HF_DIALOGUE_PLAN")]
    seg_out = [("images", "IMAGE"), ("audio", "AUDIO"), ("segment", "HF_SPEAKER_SEGMENT"), ("report", "STRING")]
    paste_in = [("images", "IMAGE"), ("corrected_images", "IMAGE"), ("segment", "HF_SPEAKER_SEGMENT")]
    sync = [("images", "IMAGE"), ("audio", "AUDIO")]
    for n, (seg, fix, paste, who, y) in enumerate(((30, 31, 32, "Woman", 760), (33, 34, 35, "Man", 1120))):
        g.add(seg, "HyperFlowSpeakerSegment", (1860, y),
              [who, "full_frame", 0.0, 0.0, 1.0, 1.0, 0.25, "final", 25.0], seg_in, seg_out, size=(320, 320),
              title=f"Segment · {who}")
        g.add(fix, "LatentSyncNode", (2220, y), [1247, 1.5, 20], sync, sync, size=(300, 150),
              title=f"LatentSync · {who}")
        g.add(paste, "HyperFlowSpeakerPaste", (2560, y), [12], paste_in, [("images", "IMAGE"), ("report", "STRING")],
              size=(300, 110), title=f"Paste · {who}")
        g.link(13 if n == 0 else 32, 0, seg, "images")
        g.link(24, 4, seg, "plan")
        g.link(seg, 0, fix, "images")
        g.link(seg, 1, fix, "audio")
        g.link(13 if n == 0 else 32, 0, paste, "images")
        g.link(fix, 0, paste, "corrected_images")
        g.link(seg, 2, paste, "segment")
    g.add(36, "CreateVideo", (2560, 1300), [24, 8, "sRGB"], [("images", "IMAGE"), ("audio", "AUDIO")],
          [("VIDEO", "VIDEO")], size=(260, 100), title="Corrected · same exact dialogue")
    g.add(37, "SaveVideo", (2560, 1440), ["video/hyperflow_ref2va_2speaker_lipsync", "auto", "auto"],
          [("video", "VIDEO")], size=(320, 320))
    g.link(35, 0, 36, "images")
    g.link(25, 1, 36, "audio")
    g.link(36, 0, 37, "video")
    for i in range(30, 38):
        g.by_id[i]["mode"] = 4

    g.add(17, "MarkdownNote", (1240, 760), [
        "# HyperFlow 8-step · two-speaker dialogue, audio-locked\n\n"
        "Each character is bound to their own line: **Woman = <Picture 1> + <Audio 1>, 0.00-7.50 s**; **Man = "
        "<Picture 2> + <Audio 2>, 7.50-15.00 s** (`start -1` = right after the previous line).\n\n"
        "- **Dialogue Track** places the lines sample-exactly on a track exactly as long as the video (15 s -> 362 "
        "frames) and appends the timing schedule to the prompt.\n"
        "- **Audio Lock · lock_source** encodes that track into the target audio latent and masks it at 0: your "
        "dialogue drives the mouths and H3 never regenerates it. HyperFlow pins those rows (r = t).\n"
        "- **CreateVideo** muxes `mux_audio`, your exact final dialogue. Wire a clean take into a line's "
        "`final_audio` to mux that instead.\n"
        "- **If lip sync falls short**, un-bypass the six correction nodes (Ctrl+B): each character is cut out on "
        "their own frames with their own track (24->25 fps for LatentSync), corrected and pasted back. Needs "
        "ComfyUI-LatentSyncWrapper. In a two-shot, set `region = box` to each face.\n\n" + COMMON_NOTE],
        size=(560, 560), title="Read me")
    g.dump("ref2va_2speaker_dialogue")


def singularity():
    """Singularity v1.3 (ref2va fine-tune): T2V / I2V / Ref2V / V2V on one reference node, switched by group bypass."""
    g = Graph()
    base(g, SINGULARITY, "singularity")
    g.add(7, "MiniMaxH3ReferenceToVideo", (460, 200),
          ["[Shot 1] A medium close-up at eye level establishes <Picture 1> centered in frame against a softly lit "
           "studio backdrop, warm practical lights glowing behind. <Picture 1> begins still, takes a breath, then "
           "speaks directly to camera in the voice from <Audio 1>, lips matching every word, eyebrows lifting and one "
           "hand rising as the sentence builds. The camera slowly pushes in from medium close-up to close-up at a "
           "steady, gentle pace, keeping the face sharp; natural matte skin, soft HDR highlights. Soundscape: the "
           "voice from <Audio 1> over quiet room tone. No music, no subtitles.", 1344, 768, 124, "match"],
          [("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"), ("ref_images.ref_image_0", "IMAGE"),
           ("ref_videos.ref_video_0", "IMAGE"), ("ref_video_audios.ref_video_audio_0", "AUDIO"),
           ("ref_audios.ref_audio_0", "AUDIO")],
          [("positive", "CONDITIONING"), ("LATENT", "LATENT")], size=(400, 340))

    # Ref2V (on): character image + voice
    g.group("Ref2V · <Picture 1> + <Audio 1> (bypass all for T2V)", (20, 580, 760, 380), "#3f789e")
    g.add(6, "LoadImage", (40, 620), ["example.png", "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
          size=(320, 320), title="Picture 1 · character")
    g.add(19, "LoadAudio", (400, 620), ["", None, None], [], [("AUDIO", "AUDIO")], size=(320, 140),
          title="Audio 1 · voice")
    # V2V (bypassed): reference clip + its soundtrack
    g.group("V2V · <Video 1> + soundtrack (Ctrl+B to enable)", (20, 990, 760, 380), "#8a5a9e")
    g.add(21, "LoadVideo", (40, 1030), [""], outputs=[("VIDEO", "VIDEO")], size=(320, 320),
          title="Video 1 · reference clip (24 fps, 2-15 s)")
    g.add(22, "GetVideoComponents", (400, 1030), [], [("video", "VIDEO")],
          [("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT"), ("bit_depth", "COMBO"),
           ("color_space", "COMBO")], size=(260, 130))
    # I2V (bypassed): first frame anchored at frame 0
    g.group("I2V · first frame at frame 0 (Ctrl+B to enable)", (20, 1400, 760, 380), "#4f8a5a")
    g.add(18, "LoadImage", (40, 1440), ["example.png", "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
          size=(320, 320), title="First frame")
    g.add(20, "MiniMaxH3AddGuide", (400, 1440), [0],
          [("positive", "CONDITIONING"), ("vae", "VAE"), ("audio_vae", "VAE"), ("latent", "LATENT"),
           ("image", "IMAGE"), ("audio", "AUDIO")], [("positive", "CONDITIONING")], size=(300, 160),
          title="Anchor first frame")
    for i in (18, 20, 21, 22):
        g.by_id[i]["mode"] = 4

    for src, slot, dst, name in (
            (3, 0, 7, "clip"), (4, 0, 7, "vae"), (5, 0, 7, "audio_vae"),
            (6, 0, 7, "ref_images.ref_image_0"), (19, 0, 7, "ref_audios.ref_audio_0"),
            (21, 0, 22, "video"), (22, 0, 7, "ref_videos.ref_video_0"), (22, 1, 7, "ref_video_audios.ref_video_audio_0"),
            (7, 0, 20, "positive"), (7, 1, 20, "latent"), (4, 0, 20, "vae"), (5, 0, 20, "audio_vae"),
            (18, 0, 20, "image"), (20, 0, 11, "conditioning"), (7, 1, 12, "latent_image")):
        g.link(src, slot, dst, name)

    g.add(17, "MarkdownNote", (900, 740), [
        "# HyperFlow 8-step · MiniMax-H3 Singularity v1.3 · T2V / I2V / Ref2V / V2V\n\n"
        "[Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity) is a ref2va fine-tune: HDR "
        "quality and less blur, face restoration in medium/long shots, no glossy skin, stronger dynamic motion "
        "(action, martial arts), spell/VFX, facial expression and camera response. Base H3 abilities are kept.\n\n"
        "**Modes** (toggle a group's nodes with Ctrl+B; they all feed one *Reference to Video* node):\n"
        "- **Ref2V** (default): <Picture 1> character + <Audio 1> voice.\n"
        "- **T2V**: bypass the Ref2V group and drop the tags from the prompt.\n"
        "- **I2V**: enable the I2V group; the image is pinned at frame 0. Bypass Ref2V unless you also want the "
        "references.\n"
        "- **V2V**: enable the V2V group and refer to the clip as <Video 1> (its soundtrack is <Audio 1>, listed "
        "before standalone audio). 24 fps, 2-15 s; longer clips are cut to the output length.\n\n"
        "**Prompting** (Singularity spec): `[Shot 1]`, `[Shot 2] At 00:03.000 ...`; causal action chains; explicit "
        "camera verbs (push-in, tracking, orbit); describe VFX trigger, form and physical consequence.\n\n"
        "**Checkpoint:** `Minimax-h3_Singularity_ref2va_v1.3_int8` only. The `_Pruned_` int8 / w4a8 files have no "
        "`time_embedder` and the loader refuses them.\n\n"
        "**Acceleration:** HyperFlow replaces the card's `ref2v_turbo_4step` LoRA; never stack the two.\n\n"
        + COMMON_NOTE], size=(640, 620), title="Read me")
    g.dump("singularity_multimodal")


if __name__ == "__main__":
    t2va()
    fl2va()
    ref2va()
    audio2va()
    dialogue()
    singularity()
    print("wrote", sorted(f for f in os.listdir(OUT) if f.endswith(".json")))
