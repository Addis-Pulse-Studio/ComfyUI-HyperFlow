#!/usr/bin/env python3
# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""Live GPU check: queue t2v, fl2v, audio-guided fl2va and ref2va HyperFlow runs on a running ComfyUI (API format).

    COMFYUI_URL=http://127.0.0.1:8188 python tools/gpu_verify.py [t2v] [fl2v] [av_guided] [ref2va]

Stdlib only. Media come from ComfyUI/input and are named by environment variables: HF_FIRST, HF_LAST (fl2v frames),
HF_REF_IMAGE (character image for av_guided / ref2va) and HF_REF_AUDIO (voice clip). Outputs land in ComfyUI/output/hyperflow_verify/. Writes each prompt it sends
to api_<name>.json in the current directory.
"""
import json, os, sys, time, urllib.request
URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
FIRST, LAST = os.environ.get("HF_FIRST", "first.png"), os.environ.get("HF_LAST", "last.png")
REF_IMAGE, REF_AUDIO = os.environ.get("HF_REF_IMAGE", "character.png"), os.environ.get("HF_REF_AUDIO", "voice.wav")
FL2VA = "minimax\\higher\\minimax_h3_fl2va_int8_convrot.safetensors"
REF2VA = "minimax\\higher\\minimax_h3_ref2va_int8_convrot.safetensors"
HF = "minimaxh3\\minimax_h3_hyperflow_8step_v1.0.safetensors"
W, H, L, SEED = 1344, 768, 124, 42

def common(ckpt, prefix):
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": ckpt, "weight_dtype": "default"}},
        "hf": {"class_type": "HyperFlowLoRALoader", "inputs": {"model": ["unet", 0], "lora_name": HF, "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "minimax\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax\\minimax_h3_video_vae_fp16.safetensors"}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax\\minimax_h3_audio_vae_fp32.safetensors"}},
        "sig": {"class_type": "HyperFlowSigmas", "inputs": {"model": ["hf", 0]}},
        "smp": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["hf", 0], "conditioning": None}},
        "sca": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["smp", 0], "sigmas": ["sig", 0], "latent_image": None}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sca", 0], "vae": ["vae", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sca", 0], "vae": ["avae", 0]}},
        "mk": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "audio": ["adec", 0], "fps": 24.0}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["mk", 0], "filename_prefix": f"hyperflow_verify/{prefix}", "format": "auto", "format.codec": "auto"}},
        "wav": {"class_type": "SaveAudio", "inputs": {"audio": ["adec", 0], "filename_prefix": f"hyperflow_verify/{prefix}"}},
    }

def i2v(g, prompt, first=None, last=None):
    g["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": W, "height": H, "length": L}}
    if first: g["img1"] = {"class_type": "LoadImage", "inputs": {"image": first}}; g["cond"]["inputs"]["first_frame"] = ["img1", 0]
    if last: g["img2"] = {"class_type": "LoadImage", "inputs": {"image": last}}; g["cond"]["inputs"]["last_frame"] = ["img2", 0]
    g["guider"]["inputs"]["conditioning"] = ["cond", 0]; g["sca"]["inputs"]["latent_image"] = ["cond", 1]
    return g

def t2v():
    return i2v(common(FL2VA, "t2v"), "A red fox trots through a snowy pine forest at dawn, breath steaming in the cold air. The soft crunch of snow under its paws and distant birdsong.")

def fl2v():
    return i2v(common(FL2VA, "fl2v"), "The subject walks forward through the scene and turns toward the camera, gentle ambient wind and footsteps.", FIRST, LAST)

def av_guided():
    g = i2v(common(FL2VA, "av_guided"), "The person looks into the camera and speaks clearly, lips moving in sync with the voice. Quiet room tone.", REF_IMAGE)
    g["aud"] = {"class_type": "LoadAudio", "inputs": {"audio": REF_AUDIO}}
    g["guide"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["cond", 0], "latent": ["cond", 1], "audio_vae": ["avae", 0], "audio": ["aud", 0], "frame_idx": 0}}
    g["guider"]["inputs"]["conditioning"] = ["guide", 0]
    return g

def ref2va():
    g = common(REF2VA, "ref2va")
    g["img1"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMAGE}}
    g["aud"] = {"class_type": "LoadAudio", "inputs": {"audio": REF_AUDIO}}
    g["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0],
        "prompt": "The character from <Picture 1> talks to the camera in the voice from <Audio 1>, lips in sync with the words, soft studio light.",
        "width": W, "height": H, "length": L, "ref_image_size": "match",
        "ref_images.ref_image_1": ["img1", 0], "ref_audios.ref_audio_1": ["aud", 0]}}
    g["guider"]["inputs"]["conditioning"] = ["cond", 0]; g["sca"]["inputs"]["latent_image"] = ["cond", 1]
    return g

def post(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:3000]}

RUNS = {"t2v": t2v, "fl2v": fl2v, "av_guided": av_guided, "ref2va": ref2va}
for name in (sys.argv[1:] or RUNS):
    g = RUNS[name]()
    json.dump(g, open(f"api_{name}.json", "w"), indent=1)
    t0 = time.time(); r = post("/prompt", {"prompt": g, "client_id": "hyperflow-verify"})
    if "prompt_id" not in r:
        print(name, "REJECTED", json.dumps(r)[:3000], flush=True); continue
    pid = r["prompt_id"]
    while True:
        h = post(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            outs = [f for n in h[pid]["outputs"].values() for k in ("images", "audio", "videos", "gifs") for f in n.get(k, [])]
            msgs = [m for m in st.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
            print(name, st.get("status_str"), f"{time.time()-t0:.0f}s", [o.get("filename") for o in outs], json.dumps(msgs)[:3000], flush=True)
            break
        time.sleep(5)
