import argparse
import glob
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PYTHONBREAKPOINT", "0")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.environ.get("DCAT_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "dcat"))

SETTINGS = {
    "vicuna-audio": ("merge_vicuna_vision_to_audio", "multimodal-vicuna-7b-v1.5-audio-dcat"),
    "vicuna-video": ("merge_vicuna_vision_to_video", "multimodal-vicuna-7b-v1.5-video-dcat"),
    "llama-audio": ("merge_llama_3_1_vision_to_audio", "ultravox_v0_4_1_llama_3_1_8b_dcat"),
    "llama-video": ("merge_llama_3_1_vision_to_video", "llava-video-llama-3.1-8b-dcat"),
}


def resolve_model_dir(model, local_name):
    if os.path.isdir(model):
        model = os.path.abspath(model)
        if os.path.basename(model.rstrip("/")) == local_name:
            return model
        link = os.path.join(CACHE_ROOT, "models", local_name)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.islink(link):
            os.remove(link)
        if not os.path.exists(link):
            os.symlink(model, link)
        return link
    from huggingface_hub import snapshot_download
    return snapshot_download(model, local_dir=os.path.join(CACHE_ROOT, "models", local_name))


def prepare_encoders(setting, model_dir, encoder_dir):
    from huggingface_hub import snapshot_download
    encoder_dir = encoder_dir or os.environ.get("MODELCOMPOSE_ENCODER_DIR") or os.path.join(CACHE_ROOT, "encoders")
    os.makedirs(encoder_dir, exist_ok=True)
    if setting == "vicuna-audio":
        dst = os.path.join(encoder_dir, "beats", "BEATs_iter3_plus_AS2M.pt")
        if not os.path.exists(dst):
            src = os.path.join(model_dir, "encoders", "beats", "BEATs_iter3_plus_AS2M.pt")
            if not os.path.exists(src):
                sys.exit(f"BEATs checkpoint not found; place BEATs_iter3_plus_AS2M.pt at {dst}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(os.path.realpath(src), dst)
    else:
        dst = os.path.join(encoder_dir, "LanguageBind_Video_merge")
        if not (os.path.exists(os.path.join(dst, "config.json")) or os.path.isdir(os.path.join(dst, "snapshots"))):
            snapshot_download("LanguageBind/LanguageBind_Video_merge", local_dir=dst)
    os.environ["MODELCOMPOSE_ENCODER_DIR"] = encoder_dir


def build_prompt(dcat, setting, question, choices):
    if not choices:
        return question
    if setting.endswith("audio"):
        return dcat.format_mmau_prompt({"question": question, "choices": choices})
    labels = "ABCDEFGHIJ"
    return dcat.format_videomme_prompt({"question": question,
                                        "options": [f"{labels[i]}. {c}" for i, c in enumerate(choices)]})


def load_ultravox(dcat, model_dir, text_model):
    import torch
    import transformers
    from safetensors.torch import load_file
    config = transformers.AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if text_model:
        config.text_model_id = text_model
    pipe = transformers.pipeline(model=model_dir, config=config, trust_remote_code=True,
                                 device="cuda", torch_dtype=torch.bfloat16)
    for shard in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
        pipe.model.load_state_dict(load_file(shard), strict=False)
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Run a DCAT-merged model on one audio or video file.")
    parser.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local checkpoint directory.")
    parser.add_argument("--media", required=True, help="Audio (.wav/.flac/...) or video (.mp4/...) file.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--choices", nargs="*", default=None,
                        help="Optional answer options; formats the prompt like the evaluation.")
    parser.add_argument("--base", default="lmsys/vicuna-7b-v1.5",
                        help="Vicuna base LLM (repo id or path); Vicuna settings only.")
    parser.add_argument("--text-model", default=None,
                        help="Override the Ultravox text_model_id (repo id or path); llama-audio only.")
    parser.add_argument("--encoder-dir", default=None,
                        help="Directory for the ModelCompose encoders (BEATs / LanguageBind); Vicuna settings only.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    script_dir, local_name = SETTINGS[args.setting]
    sys.path.insert(0, os.path.join(REPO_ROOT, script_dir))
    import mi_surrogate_merge_eval as dcat

    model_dir = resolve_model_dir(args.model, local_name)
    prompt = build_prompt(dcat, args.setting, args.question, args.choices)

    if args.setting.startswith("vicuna"):
        prepare_encoders(args.setting, model_dir, args.encoder_dir)
        tokenizer, model, modal_processors = dcat.load_model(model_dir, args.base)
        model_dict = {"tokenizer": tokenizer, "model": model, "modal_processors": modal_processors}
        if args.setting == "vicuna-audio":
            answer = dcat.run_audio_inference(model_dict, args.media, prompt, max_new_tokens=args.max_new_tokens)
        else:
            answer = dcat.run_video_inference(model_dict, args.media, prompt, max_new_tokens=args.max_new_tokens)
    elif args.setting == "llama-audio":
        pipe = load_ultravox(dcat, model_dir, args.text_model)
        model_dict = {"tokenizer": pipe.tokenizer, "model": pipe.model, "image_processor": pipe}
        answer = dcat.run_audio_inference(model_dict, args.media, f"<|audio|>\n{prompt}",
                                          max_new_tokens=args.max_new_tokens)
    else:
        tokenizer, model, image_processor = dcat.load_model(model_dir, None)
        model_dict = {"tokenizer": tokenizer, "model": model, "image_processor": image_processor}
        answer = dcat.run_video_inference(model_dict, args.media, prompt, max_new_tokens=args.max_new_tokens)

    print("\n=== Prompt ===\n" + prompt)
    print("\n=== Answer ===\n" + answer)


if __name__ == "__main__":
    main()
