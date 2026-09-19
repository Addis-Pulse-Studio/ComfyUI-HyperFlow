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

COMMON_NOTE = """**Checkpoint:** a non-pruned H3 (`*_int8_convrot` or `*_bf16`). The `*_pruned_*` files are curve-form (no
`time_embedder`) and the loader refuses them.

**Weights:** `hf download videorebirth/hyperflow minimax_h3_hyperflow_8step_v1.0.safetensors
--local-dir models/loras/minimaxh3` (MiniMax H3 Community License; read it first).

**Sampling is fixed:** `HyperFlow Sigmas` gives the 8-step grid stored in the file; sampler `euler`;
`BasicGuider` (cfg 1.0). No steps, no scheduler, no CFG. Another schedule or a multi-stage sampler raises.

Don't stack HyperFlow with FastH3 / Turbo / TaoMate step LoRAs."""


class Graph:
    def __init__(self):
        self.nodes, self.links, self.by_id = [], [], {}

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
        lid = len(self.links) + 1
        s, d = self.by_id[src], self.by_id[dst]
        kind = s["outputs"][slot]["type"]
        dslot = next(j for j, x in enumerate(d["inputs"]) if x["name"] == name)
        s["outputs"][slot]["links"].append(lid)
        d["inputs"][dslot]["link"] = lid
        self.links.append([lid, src, slot, dst, dslot, kind])

    def dump(self, name):
        wf = {"id": f"hyperflow-{name}", "revision": 0, "last_node_id": max(self.by_id), "last_link_id": len(self.links),
              "nodes": self.nodes, "links": self.links, "groups": [], "config": {},
              "extra": {"ds": {"scale": 0.75, "offset": [0, 0]}}, "version": 0.4}
        with open(os.path.join(OUT, f"hyperflow_{name}.json"), "w", encoding="utf-8") as handle:
            json.dump(wf, handle, indent=1, ensure_ascii=False)
            handle.write("\n")


def base(g: Graph, checkpoint: str, prefix: str):
    """Loaders, HyperFlow, sampler and decode/save. Conditioning node 7 is added by the caller."""
    g.add(1, "UNETLoader", (40, 60), [checkpoint, "default"], outputs=[("MODEL", "MODEL")], size=(380, 90))
    g.add(2, "HyperFlowLoRALoader", (460, 60), [HYPERFLOW, 1.0], [("model", "MODEL")], [("MODEL", "MODEL")], size=(380, 90))
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


def image_to_video(g: Graph, prompt: str):
    g.add(7, "MiniMaxH3ImageToVideo", (460, 200), [prompt, 1344, 768, 124],
          [("clip", "CLIP"), ("vae", "VAE"), ("first_frame", "IMAGE"), ("last_frame", "IMAGE")],
          [("positive", "CONDITIONING"), ("LATENT", "LATENT")], size=(400, 300))
    g.link(3, 0, 7, "clip")
    g.link(4, 0, 7, "vae")
    g.link(7, 1, 12, "latent_image")


def t2v():
    g = Graph()
    base(g, FL2VA, "t2v")
    image_to_video(g, "A red fox trots through a snowy pine forest at dawn, breath steaming in the cold air. "
                      "The soft crunch of snow under its paws and distant birdsong.")
    g.link(7, 0, 11, "conditioning")
    g.add(17, "MarkdownNote", (460, 560), ["# HyperFlow 8-step · MiniMax H3 text → video + audio\n\n"
                                           "t2va runs on the fl2va checkpoint with no frames connected.\n\n" + COMMON_NOTE],
          size=(520, 380), title="Read me")
    g.dump("t2v")


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
           ("ref_audios.ref_audio_0", "AUDIO")],
          [("positive", "CONDITIONING"), ("LATENT", "LATENT")], size=(400, 320))
    g.link(3, 0, 7, "clip")
    g.link(4, 0, 7, "vae")
    g.link(5, 0, 7, "audio_vae")
    g.link(6, 0, 7, "ref_images.ref_image_0")
    g.link(19, 0, 7, "ref_audios.ref_audio_0")
    g.link(7, 0, 11, "conditioning")
    g.link(7, 1, 12, "latent_image")
    g.add(17, "MarkdownNote", (460, 580), ["# HyperFlow 8-step · MiniMax H3 reference image + audio → video + audio\n\n"
                                           "Avatar / character animation on the **ref2va** checkpoint. The same "
                                           "HyperFlow file works: it never touches `adaln_proj`, which carries the "
                                           "reference conditioning. Reference rows stay pinned (r = t).\n\n" + COMMON_NOTE],
          size=(520, 400), title="Read me")
    g.dump("ref2va")


if __name__ == "__main__":
    t2v()
    fl2va()
    ref2va()
    print("wrote", sorted(f for f in os.listdir(OUT) if f.endswith(".json")))
