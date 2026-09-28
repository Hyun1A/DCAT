#!/usr/bin/env python3

import os
import sys
import gc
import json
import time
import argparse
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
MODELCOMPOSE_DIR = os.path.join(SRC_DIR, "MLLMerging", "ModelCompose")
if MODELCOMPOSE_DIR not in sys.path:
    sys.path.insert(0, MODELCOMPOSE_DIR)

DEFAULT_MMAU_ROOT = "/dev/shm/mmau"
DEFAULT_VIDEOMME_ROOT = "/dev/shm/videomme"
DEFAULT_CACHE_DIR = "/dev/shm/modal_cache"
DEFAULT_MODEL_BASE = "xxx/xxx/vicuna_models/checkpoints/vicuna-7b-v1.5"


def load_processors(ckpt_path, model_base):
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    from modelcompose.model.builder import load_pretrained_model
    from modelcompose.utils import disable_torch_init
    from modelcompose.mm_utils import get_model_name_from_path
    disable_torch_init()
    model_name = get_model_name_from_path(ckpt_path)
    print(f"  Loading: {ckpt_path}")
    tokenizer, model, modal_processors, _ = load_pretrained_model(
        ckpt_path, model_base, model_name
    )
    return model, modal_processors


def collect_mmau_audio_paths(mmau_root):
    json_path = os.path.join(mmau_root, "test_mini.json")
    if not os.path.isfile(json_path):
        print(f"  [MMAU] JSON not found: {json_path}")
        return []
    with open(json_path) as f:
        data = json.load(f)
    paths = []
    for item in data:
        p = item.get("audio_path")
        if not p:
            continue
        if not os.path.isabs(p):
            p = os.path.join(mmau_root, p)
        if os.path.isfile(p):
            paths.append(p)
    return list(dict.fromkeys(paths))


def collect_videomme_video_paths(videomme_root):
    json_path = os.path.join(videomme_root, "videomme_test_flat.json")
    video_dir = os.path.join(videomme_root, "videos", "data")
    if not os.path.isfile(json_path):
        print(f"  [VideoMME] JSON not found: {json_path}")
        return []
    with open(json_path) as f:
        data = json.load(f)
    paths = []
    for item in data:
        vid = item.get("videoID", "")
        p = os.path.join(video_dir, f"{vid}.mp4")
        if os.path.isfile(p):
            paths.append(p)
    return list(dict.fromkeys(paths))


def preload_audio(audio_ckpt, model_base, mmau_root, cache_dir):
    paths = collect_mmau_audio_paths(mmau_root)
    if not paths:
        print("  [MMAU] No audio paths collected; skipping")
        return 0
    out_path = os.path.join(cache_dir, "audio.pt")
    if os.path.isfile(out_path):
        print(f"  [MMAU] cache already exists at {out_path} — skipping preload")
        return 0

    print(f"  [MMAU] Loading source-audio checkpoint for audio processor...")
    model, modal_processors = load_processors(audio_ckpt, model_base)
    if modal_processors is None or "audio" not in modal_processors:
        print(f"  [MMAU] ERROR: checkpoint has no audio processor. Abort.")
        return 0
    audio_proc = modal_processors["audio"]

    print(f"  [MMAU] Decoding {len(paths)} audio clips into RAM cache...")
    cache = {}
    t0 = time.time()
    for p in tqdm(paths, desc="  audio"):
        try:
            cache[p] = audio_proc([p])
        except Exception as e:
            print(f"    [MMAU] FAIL {p}: {e}")
    dt = time.time() - t0
    print(f"  [MMAU] Decoded {len(cache)}/{len(paths)} in {dt:.1f}s")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(cache, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  [MMAU] Saved {out_path} ({size_mb:.1f} MB)")

    del model, modal_processors, audio_proc, cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 1


def preload_video(video_ckpt, model_base, videomme_root, cache_dir):
    paths = collect_videomme_video_paths(videomme_root)
    if not paths:
        print("  [VideoMME] No video paths collected; skipping")
        return 0
    out_path = os.path.join(cache_dir, "video.pt")
    if os.path.isfile(out_path):
        print(f"  [VideoMME] cache already exists at {out_path} — skipping preload")
        return 0

    print(f"  [VideoMME] Loading source-video checkpoint for video processor...")
    model, modal_processors = load_processors(video_ckpt, model_base)
    if modal_processors is None or "video" not in modal_processors:
        print(f"  [VideoMME] ERROR: checkpoint has no video processor. Abort.")
        return 0
    video_proc = modal_processors["video"]

    print(f"  [VideoMME] Decoding {len(paths)} video clips into RAM cache...")
    cache = {}
    t0 = time.time()
    for p in tqdm(paths, desc="  video"):
        try:
            out = video_proc(p)
            if isinstance(out, dict):
                cache[p] = {k: v for k, v in out.items()}
            else:
                cache[p] = {"pixel_values": out}
        except Exception as e:
            print(f"    [VideoMME] FAIL {p}: {e}")
    dt = time.time() - t0
    print(f"  [VideoMME] Decoded {len(cache)}/{len(paths)} in {dt:.1f}s")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(cache, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  [VideoMME] Saved {out_path} ({size_mb:.1f} MB)")

    del model, modal_processors, video_proc, cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 1


def main():
    parser = argparse.ArgumentParser(description="Preload MMAU + VideoMME into /dev/shm")
    parser.add_argument("--audio-checkpoint", type=str, default="",
                        help="Checkpoint with audio processor (source audio model). "
                             "If empty, audio cache is skipped.")
    parser.add_argument("--video-checkpoint", type=str, default="",
                        help="Checkpoint with video processor (source video model). "
                             "If empty, video cache is skipped.")
    parser.add_argument("--model-base", type=str, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--mmau-root", type=str, default=DEFAULT_MMAU_ROOT)
    parser.add_argument("--videomme-root", type=str, default=DEFAULT_VIDEOMME_ROOT)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    t_all = time.time()
    os.makedirs(args.cache_dir, exist_ok=True)
    print("=" * 70)
    print(f"  Modal-cache preload into {args.cache_dir}")
    print("=" * 70)

    audio_ok = True
    video_ok = True

    if not args.skip_audio and args.audio_checkpoint:
        rc = preload_audio(args.audio_checkpoint, args.model_base, args.mmau_root, args.cache_dir)
        audio_ok = bool(rc) or os.path.isfile(os.path.join(args.cache_dir, "audio.pt"))
        if not audio_ok:
            print("  [MMAU] preload produced no audio.pt — refusing to write READY")
    else:
        print("  [MMAU] skipped")

    if not args.skip_video and args.video_checkpoint:
        rc = preload_video(args.video_checkpoint, args.model_base, args.videomme_root, args.cache_dir)
        video_ok = bool(rc) or os.path.isfile(os.path.join(args.cache_dir, "video.pt"))
        if not video_ok:
            print("  [VideoMME] preload produced no video.pt — refusing to write READY")
    else:
        print("  [VideoMME] skipped")

    all_ok = audio_ok and video_ok
    if all_ok:
        ready_path = os.path.join(args.cache_dir, "READY")
        with open(ready_path, "w") as f:
            f.write(f"ready @ {time.time()}\n")
        marker_msg = f"marker: {ready_path}"
    else:
        marker_msg = "marker: NOT WRITTEN (preload incomplete)"
    print("=" * 70)
    print(f"  Done in {time.time()-t_all:.1f}s | {marker_msg}")
    print("=" * 70)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
