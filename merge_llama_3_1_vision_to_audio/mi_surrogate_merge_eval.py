#!/usr/bin/env python3

import os
import sys
import gc
import re
import json
import time
import argparse
import logging
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR_MMAU = "/dev/shm/mmau"
AVQA_DATA_ROOT = "/dev/shm/avqa_data/data"
AVQA_VIDEO_CONSTRUCTED_DIR = "xxx/xxx/data/AVQA/constructed_avqa/videos/test"
VIDEOMME_DATA = "/dev/shm/videomme"
VIDEOMME_VIDEO_DIR = os.path.join(VIDEOMME_DATA, "videos", "data")
VIDEOMME_FLAT_JSON = os.path.join(VIDEOMME_DATA, "videomme_test_flat.json")

NUM_LAYERS = 32
ACCUMULATION_DTYPE = torch.float32

PROJ_GROUP_MAP_FULL = {
    'attn_full': ['self_attn.q_proj', 'self_attn.k_proj',
                  'self_attn.v_proj', 'self_attn.o_proj'],
    'mlp_full':  ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
}

PROJ_DEPENDENCY_GROUPS = [
    ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
    ['self_attn.o_proj'],
    ['mlp.gate_proj', 'mlp.up_proj'],
    ['mlp.down_proj'],
]


def split_into_dependency_groups(proj_paths):
    proj_set = set(proj_paths)
    groups = []
    for dep_group in PROJ_DEPENDENCY_GROUPS:
        filtered = [p for p in dep_group if p in proj_set]
        if filtered:
            groups.append(filtered)
    return groups


def resolve_proj_paths(proj_groups_str):
    tokens = [t.strip() for t in proj_groups_str.split(',') if t.strip()]
    expanded = []
    for t in tokens:
        if t == 'all':
            expanded.extend(['attn_full', 'mlp_full'])
        elif t in PROJ_GROUP_MAP_FULL:
            expanded.append(t)
        else:
            raise ValueError(
                f"Unsupported proj group '{t}'. "
                f"Allowed: attn_full, mlp_full, all.")
    seen = set()
    groups = []
    for g in expanded:
        if g not in seen:
            seen.add(g); groups.append(g)
    paths = []
    for g in groups:
        paths.extend(PROJ_GROUP_MAP_FULL[g])
    return groups, paths


def _patch_ultravox_config(model_path, llama_base_path):
    config_json = os.path.join(model_path, "config.json")
    if not os.path.exists(config_json): return
    with open(config_json) as f: cfg = json.load(f)
    text_model_id = cfg.get("text_model_id", "")
    if text_model_id and os.path.isdir(llama_base_path):
        cfg["text_model_id"] = llama_base_path
        patched_config = os.path.join(model_path, "config.json.bak")
        if not os.path.exists(patched_config):
            import shutil; shutil.copy2(config_json, patched_config)
        with open(config_json, 'w') as f: json.dump(cfg, f, indent=2)

def _restore_config(model_path):
    import shutil
    config_json = os.path.join(model_path, "config.json")
    patched_config = os.path.join(model_path, "config.json.bak")
    if os.path.exists(patched_config):
        shutil.move(patched_config, config_json)

def load_model(ckpt_path, model_base):
    import transformers
    print(f"  Loading Ultravox model via pipeline: {ckpt_path}")
    if model_base: _patch_ultravox_config(ckpt_path, model_base)
    t0 = time.time()
    try:
        pipe = transformers.pipeline(
            model=ckpt_path, trust_remote_code=True,
            device="cuda", torch_dtype=torch.bfloat16,
        )
    finally:
        if model_base: _restore_config(ckpt_path)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    model = pipe.model
    model.model = model.language_model.model
    tokenizer = pipe.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, pipe


def load_safetensors_weights(model_path):
    from safetensors.torch import load_file as load_safetensors
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index['weight_map'].values()))
        all_weights = {}
        for sf in shard_files:
            fp = os.path.join(model_path, sf)
            print(f"    Loading shard: {sf}")
            all_weights.update(load_safetensors(fp))
        return all_weights
    single_files = sorted(f for f in os.listdir(model_path)
                          if f.endswith('.safetensors'))
    if single_files:
        all_weights = {}
        for sf in single_files:
            all_weights.update(load_safetensors(os.path.join(model_path, sf)))
        return all_weights
    raise FileNotFoundError(f"No safetensors files found in {model_path}")


def load_base_model_weights(model_base_path):
    index_st = os.path.join(model_base_path, 'model.safetensors.index.json')
    if os.path.exists(index_st):
        return load_safetensors_weights(model_base_path)
    index_path = os.path.join(model_base_path, 'pytorch_model.bin.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = set(index['weight_map'].values())
        all_weights = {}
        for shard_file in sorted(shard_files):
            shard_path = os.path.join(model_base_path, shard_file)
            print(f"    Loading shard: {shard_file}")
            shard_weights = torch.load(shard_path, map_location='cpu',
                                       weights_only=False)
            all_weights.update(shard_weights)
            del shard_weights
        return all_weights
    single_path = os.path.join(model_base_path, 'pytorch_model.bin')
    if os.path.exists(single_path):
        return torch.load(single_path, map_location='cpu', weights_only=False)
    raise FileNotFoundError(f"No model weights found in {model_base_path}")


def _detect_llm_prefix(state_dict):
    marker = "layers.0.self_attn.q_proj.weight"
    for key in state_dict:
        if key.endswith(marker) and "vision_tower" not in key \
                and "audio_tower" not in key and "video_tower" not in key:
            prefix = key[: -len(marker)]
            return prefix
    return ""


def load_donor_weights(donor_path):
    raw = load_safetensors_weights(donor_path)
    prefix = _detect_llm_prefix(raw)
    canonical_prefix = "model."
    if prefix == canonical_prefix:
        return raw
    print(f"    Remapping donor keys: '{prefix}' → '{canonical_prefix}'")
    remapped = {}
    for k, v in raw.items():
        if k.startswith(prefix):
            new_key = canonical_prefix + k[len(prefix):]
            remapped[new_key] = v
        else:
            remapped[k] = v
    return remapped


def get_effective_weight(base_weights, donor_weights, layer_idx, proj_path,
                         lora_alpha=None, lora_r=None):
    key = f'model.layers.{layer_idx}.{proj_path}.weight'
    if key in donor_weights:
        return donor_weights[key].float()
    if key in base_weights:
        return base_weights[key].float()
    raise KeyError(f"Weight key not found: {key}")


def get_current_effective_weight(model, layer_idx, proj_path):
    layer = model.model.layers[layer_idx]
    module = layer
    for part in proj_path.split('.'):
        module = getattr(module, part)
    return module.weight.data.float()


def update_model_weight_inplace(model, layer_idx, proj_path, W_new):
    layer = model.model.layers[layer_idx]
    module = layer
    for part in proj_path.split('.'):
        module = getattr(module, part)
    W_new_dev = W_new.to(dtype=model.dtype, device=model.device)
    module.weight.data.copy_(W_new_dev)


def format_mmau_prompt(sample):
    question = sample.get("instruction", sample.get("question", ""))
    choices = sample.get("choices", [])
    if isinstance(choices, str):
        choices = json.loads(choices)
    labels = "ABCDEFGHIJ"
    choices_text = "\n".join(f"({labels[i]}) {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{choices_text}\n\n"
            "Answer with the correct choice. Only provide the answer text, "
            "do not include the letter label.")


def load_mmau_audio_data(data_root, variant="test-mini", testset="full", num_samples=256):
    variant_key = variant.replace("-", "_")
    json_path = os.path.join(data_root, f"{variant_key}.json")
    with open(json_path) as f:
        all_data = json.load(f)
    print(f"  MMAU raw: {len(all_data)} samples")
    valid_data = []
    for item in all_data:
        if 'audio_path' not in item:
            continue
        p = item['audio_path']
        if not os.path.isabs(p):
            p = os.path.join(data_root, p)
        if os.path.exists(p):
            item['audio_path'] = p
            valid_data.append(item)
    if testset != "full":
        limits = {"medium": 1000, "small": 300, "tiny": 100}
        import random
        rng = random.Random(42)
        indices = list(range(len(valid_data)))
        rng.shuffle(indices)
        valid_data = [valid_data[i] for i in indices[:limits.get(testset, 1000)]]
    if len(valid_data) > num_samples:
        import random
        rng = random.Random(42)
        idx = rng.sample(range(len(valid_data)), num_samples)
        idx.sort()
        valid_data = [valid_data[i] for i in idx]
    for item in valid_data:
        item['modal_inputs'] = {'audio': [item['audio_path']]}
        if 'conversations' not in item:
            item['conversations'] = [{'value': f"<|audio|>\n{format_mmau_prompt(item)}"}]
    return valid_data


def load_avqa_audio_data(data_root, num_samples=256):
    json_path = os.path.join(data_root, 'test', 'avqa-test_mm_video+image+audio.json')
    with open(json_path) as f:
        all_data = json.load(f)
    valid = []
    for item in all_data:
        modal_inputs = item.get('modal_inputs', {})
        if 'audio' not in modal_inputs:
            continue
        audio_paths = modal_inputs['audio']
        ok = True
        for p in audio_paths:
            fp = os.path.join(data_root, 'evaluation_datasets',
                              p.replace('data/evaluation_datasets/', ''))
            if not os.path.exists(fp):
                ok = False; break
        if ok:
            valid.append(item)
    if len(valid) > num_samples:
        import random
        rng = random.Random(42)
        idx = rng.sample(range(len(valid)), num_samples)
        idx.sort()
        valid = [valid[i] for i in idx]
    for item in valid:
        item['modal_inputs'] = {'audio': item['modal_inputs']['audio']}
        abs_paths = []
        for p in item['modal_inputs']['audio']:
            if not os.path.isabs(p):
                abs_paths.append(os.path.join(data_root, 'evaluation_datasets',
                                              p.replace('data/evaluation_datasets/', '')))
            else:
                abs_paths.append(p)
        item['modal_inputs']['audio'] = abs_paths
        conv_text = item['conversations'][0]['value']
        conv_text = re.sub(r'Video:\s*<video>\s*\n?\s*', '', conv_text)
        conv_text = re.sub(r'Image:\s*<image>\s*\n?\s*', '', conv_text)
        item['conversations'][0]['value'] = conv_text.strip()
    return valid


def load_avqa_video_data(num_samples=128):
    json_path = os.path.join(AVQA_DATA_ROOT, 'test', 'avqa-test_mm_video.json')
    with open(json_path) as f:
        all_data = json.load(f)
    valid = []
    for item in all_data:
        modal_inputs = item.get('modal_inputs', {})
        if 'video' not in modal_inputs:
            continue
        patched = []
        ok = True
        for p in modal_inputs['video']:
            fname = os.path.basename(p)
            full_p = os.path.join(AVQA_VIDEO_CONSTRUCTED_DIR, fname)
            if not os.path.exists(full_p):
                ok = False; break
            patched.append(full_p)
        if ok:
            new = dict(item)
            new['modal_inputs'] = {'video': patched}
            conv_text = new['conversations'][0]['value']
            conv_text = re.sub(r'Audio:\s*<audio>\s*\n?\s*', '', conv_text)
            conv_text = re.sub(r'Image:\s*<image>\s*\n?\s*', '', conv_text)
            new['conversations'] = list(new['conversations'])
            new['conversations'][0] = dict(new['conversations'][0])
            new['conversations'][0]['value'] = conv_text.strip()
            valid.append(new)
    if len(valid) > num_samples:
        import random
        rng = random.Random(42)
        idx = list(range(len(valid))); rng.shuffle(idx)
        valid = [valid[i] for i in sorted(idx[:num_samples])]
    return valid


def load_videomme_eval_data(num_samples=0, seed=42):
    with open(VIDEOMME_FLAT_JSON) as f:
        data = json.load(f)
    valid = []
    for item in data:
        vid = item.get("videoID", "")
        vp = os.path.join(VIDEOMME_VIDEO_DIR, f"{vid}.mp4")
        if os.path.exists(vp):
            item["video_path"] = vp
            valid.append(item)
    if num_samples > 0 and num_samples < len(valid):
        import random
        rng = random.Random(seed)
        by_dur = {"short": [], "medium": [], "long": []}
        for item in valid:
            by_dur.get(item.get("duration", "short"), by_dur["short"]).append(item)
        per = num_samples // 3
        rem = num_samples - per * 3
        sampled = []
        for i, (dur, items) in enumerate(sorted(by_dur.items())):
            n = per + (1 if i < rem else 0)
            n = min(n, len(items))
            rng.shuffle(items)
            sampled.extend(items[:n])
        rng.shuffle(sampled)
        valid = sampled
    return valid


def extract_initial_hidden_states(model, tokenizer, pipe, data,
                                   start_layer, modality='audio'):
    assert modality in ('audio',), f"Only audio modality supported, got {modality}"
    import librosa

    processor = pipe.processor

    captured_states = []
    layer_input_captured = {}

    def layer_input_hook(module, args, kwargs=None):
        if 'hs' not in layer_input_captured:
            if isinstance(args, tuple) and len(args) > 0:
                hs = args[0]
                if hs.shape[1] > 1:
                    layer_input_captured['hs'] = hs.detach()
        return None

    h = model.model.layers[start_layer].register_forward_pre_hook(
        layer_input_hook, with_kwargs=True)

    device = next(model.parameters()).device

    for idx, item in enumerate(tqdm(data, desc="  Phase1 hs", leave=False)):
        try:
            raw = item.get('modal_inputs', {})
            if 'audio' in raw:
                paths = raw['audio']
                audio_path = paths[0] if isinstance(paths, list) else paths
            elif 'audio_path' in item:
                audio_path = item['audio_path']
            else:
                continue

            audio, _ = librosa.load(audio_path, sr=16000)
            prompt_text = item['conversations'][0]['value']

            layer_input_captured.clear()

            inputs = processor(text=prompt_text, audio=audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                _ = pipe.model(**inputs, output_hidden_states=False)

            if 'hs' not in layer_input_captured:
                continue

            hs = layer_input_captured['hs'][0]

            if "audio_token_start_idx" not in inputs or "audio_token_len" not in inputs:
                continue

            fip = inputs["audio_token_start_idx"][0].item()
            num_audio_tokens = inputs["audio_token_len"][0].item()

            if num_audio_tokens <= 0:
                continue

            H_src = hs[fip : fip + num_audio_tokens].contiguous()
            H_t = torch.cat([
                hs[:fip],
                hs[fip + num_audio_tokens:]
            ], dim=0).contiguous()

            if H_src.shape[0] == 0:
                continue
            captured_states.append({'H_src': H_src, 'H_text': H_t})
        except Exception as e:
            if idx < 3:
                print(f"    WARNING: sample {idx} failed: {e}")
            continue

    h.remove()
    return captured_states


def stack_features_for_layer(model, layer_idx, proj_paths, samples, device,
                             capture_residuals=False):
    layer = model.model.layers[layer_idx]

    resid_paths = [p for p in proj_paths
                   if capture_residuals and (p.endswith('o_proj') or
                                             p.endswith('down_proj'))]
    out = {p: {'G_s': None, 'G_t': None, 'N_s': 0, 'N_t': 0} for p in proj_paths}
    for p in resid_paths:
        out[p].update({'B_s': None, 'B_t': None, 'D_s': None, 'D_t': None})

    captured = {}
    residual_attn = {'value': None}

    def hook_factory(key):
        def hook(module, args, kwargs=None):
            if isinstance(args, tuple) and len(args) > 0:
                X = args[0]
            elif kwargs and 'hidden_states' in kwargs:
                X = kwargs['hidden_states']
            else:
                X = args[0] if isinstance(args, tuple) else args
            captured[key] = X.detach()
        return hook

    def residual_attn_hook(module, args, kwargs=None):
        if isinstance(args, tuple) and len(args) > 0:
            X = args[0]
        elif kwargs and 'hidden_states' in kwargs:
            X = kwargs['hidden_states']
        else:
            X = args[0] if isinstance(args, tuple) else args
        residual_attn['value'] = X.detach()

    handles = []
    for pp in proj_paths:
        mod = layer
        for part in pp.split('.'):
            mod = getattr(mod, part)
        handles.append(mod.register_forward_pre_hook(hook_factory(pp), with_kwargs=True))

    need_attn_residual = any(p.endswith('down_proj') for p in resid_paths)
    if need_attn_residual:
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            residual_attn_hook, with_kwargs=True))

    def _accumulate(bucket_key, stream_key):
        for pp in proj_paths:
            if pp not in captured:
                continue
            X = captured[pp][0].to(torch.float32)
            XTX = X.T @ X
            if out[pp][bucket_key] is None:
                out[pp][bucket_key] = XTX
            else:
                out[pp][bucket_key] += XTX
            out[pp][stream_key] += X.shape[0]

    def _accumulate_residuals(H_layer_input, stream):
        if not resid_paths:
            return
        R_layer = H_layer_input[0].to(torch.float32)
        R_attn = None
        if need_attn_residual and residual_attn['value'] is not None:
            R_attn = residual_attn['value'][0].to(torch.float32)
        for pp in resid_paths:
            if pp not in captured:
                continue
            if pp.endswith('o_proj'):
                R = R_layer
            elif pp.endswith('down_proj'):
                if R_attn is None:
                    continue
                R = R_attn
            else:
                continue
            X = captured[pp][0].to(torch.float32)
            B_sum = R.T @ X
            D_sum = R.T @ R
            b_key = 'B_s' if stream == 's' else 'B_t'
            d_key = 'D_s' if stream == 's' else 'D_t'
            out[pp][b_key] = B_sum if out[pp][b_key] is None else out[pp][b_key] + B_sum
            out[pp][d_key] = D_sum if out[pp][d_key] is None else out[pp][d_key] + D_sum

    try:
        for sample in samples:
            H_src_cached = sample['H_src']
            H_t_cached = sample['H_text']
            H_src = H_src_cached.unsqueeze(0).to(device=device, dtype=model.dtype,
                                                 non_blocking=True)
            H_t = H_t_cached.unsqueeze(0).to(device=device, dtype=model.dtype,
                                             non_blocking=True)
            captured.clear()
            residual_attn['value'] = None
            with torch.inference_mode():
                pos = torch.arange(H_src.shape[1], device=device).unsqueeze(0)
                pos_emb = model.model.rotary_emb(H_src, pos)
                _ = layer(H_src, position_ids=pos, position_embeddings=pos_emb)
            _accumulate('G_s', 'N_s')
            _accumulate_residuals(H_src, 's')
            captured.clear()
            residual_attn['value'] = None
            with torch.inference_mode():
                pos_t = torch.arange(H_t.shape[1], device=device).unsqueeze(0)
                pos_emb_t = model.model.rotary_emb(H_t, pos_t)
                _ = layer(H_t, position_ids=pos_t, position_embeddings=pos_emb_t)
            _accumulate('G_t', 'N_t')
            _accumulate_residuals(H_t, 't')
    finally:
        for h in handles:
            h.remove()

    for pp in proj_paths:
        if out[pp]['G_s'] is not None and out[pp]['N_s'] > 0:
            out[pp]['G_s'] /= out[pp]['N_s']
        if out[pp]['G_t'] is not None and out[pp]['N_t'] > 0:
            out[pp]['G_t'] /= out[pp]['N_t']
        if pp in resid_paths:
            if out[pp]['B_s'] is not None and out[pp]['N_s'] > 0:
                out[pp]['B_s'] /= out[pp]['N_s']
            if out[pp]['B_t'] is not None and out[pp]['N_t'] > 0:
                out[pp]['B_t'] /= out[pp]['N_t']
            if out[pp]['D_s'] is not None and out[pp]['N_s'] > 0:
                out[pp]['D_s'] /= out[pp]['N_s']
            if out[pp]['D_t'] is not None and out[pp]['N_t'] > 0:
                out[pp]['D_t'] /= out[pp]['N_t']
    return out


def propagate_through_layer(model, layer_idx, samples, device):
    layer = model.model.layers[layer_idx]
    for sample in samples:
        H_src = sample['H_src'].unsqueeze(0).to(device=device, dtype=model.dtype,
                                                non_blocking=True)
        H_t = sample['H_text'].unsqueeze(0).to(device=device, dtype=model.dtype,
                                               non_blocking=True)
        with torch.inference_mode():
            pos_s = torch.arange(H_src.shape[1], device=device).unsqueeze(0)
            pos_emb_s = model.model.rotary_emb(H_src, pos_s)
            out_s = layer(H_src, position_ids=pos_s, position_embeddings=pos_emb_s)
            if isinstance(out_s, tuple):
                out_s = out_s[0]
            sample['H_src'] = out_s[0].detach()
            pos_t = torch.arange(H_t.shape[1], device=device).unsqueeze(0)
            pos_emb_t = model.model.rotary_emb(H_t, pos_t)
            out_t = layer(H_t, position_ids=pos_t, position_embeddings=pos_emb_t)
            if isinstance(out_t, tuple):
                out_t = out_t[0]
            sample['H_text'] = out_t[0].detach()


def _sym_matrix_inv_sqrt(M, eps_rel=1e-10):
    M = 0.5 * (M + M.T)
    evals, evecs = torch.linalg.eigh(M)
    evals_c = evals.clamp(min=0)
    max_e = evals_c.max().clamp(min=1e-30)
    thresh = (max_e * eps_rel).item()
    inv_sqrt = torch.where(evals_c > thresh,
                           1.0 / torch.sqrt(evals_c.clamp(min=1e-30)),
                           torch.zeros_like(evals_c))
    return (evecs * inv_sqrt.unsqueeze(0)) @ evecs.T


def compute_tau_m_mi_surrogate(
    W_base, W_r, donor_weights_list,
    G_mod, G_text,
    eta, top_p,
    device='cpu', eps_solve=1e-6,
    alpha=0.5,
    beta=0.5,
    beta_rank=None,
    fixed_alpha=False,
    fixed_beta=False,
):
    alpha_fallback = float(alpha)
    alpha = float(alpha)
    beta_fallback = float(beta)
    if beta_rank is None:
        beta_rank = int(top_p)

    W_base = W_base.to(device).to(torch.float32)
    W_r = W_r.to(device).to(torch.float32)
    donor_W = [W.to(device).to(torch.float32) for W in donor_weights_list]
    G_mod = G_mod.to(device).to(torch.float32)
    G_text = G_text.to(device).to(torch.float32)
    d_out, d_in = W_r.shape

    tau_r = W_r - W_base
    tau_d_list = [W - W_base for W in donor_W]
    tau_d = tau_d_list[0] if tau_d_list else torch.zeros_like(tau_r)

    YtYt = W_r @ G_text @ W_r.T
    YtYt = 0.5 * (YtYt + YtYt.T)
    eigvals_t, eigvecs_t = torch.linalg.eigh(YtYt)
    p_eff = int(min(top_p, d_out))
    V_text = eigvecs_t[:, -p_eff:]

    G_Yr = W_r @ G_mod @ W_r.T
    G_Yr = 0.5 * (G_Yr + G_Yr.T)
    norm_Yr_sq = torch.trace(G_Yr).clamp(min=0).item()
    norm_Yr = norm_Yr_sq ** 0.5
    rho = norm_Yr / max(p_eff, 1) ** 0.5

    M_p = V_text.T @ G_Yr @ V_text
    M_p = 0.5 * (M_p + M_p.T)
    eig_M_p, U_0 = torch.linalg.eigh(M_p)
    sigma_sq = eig_M_p.clamp(min=0)
    sigma = torch.sqrt(sigma_sq.clamp(min=1e-30))

    SEO_r = float(sigma_sq.sum().item()) / max(norm_Yr_sq, 1e-30)
    SEO_r = min(max(SEO_r, 1e-12), 1.0)

    _p_total = sigma_sq.sum().clamp(min=1e-30)
    _p_norm = sigma_sq / _p_total
    _p_nz = _p_norm[_p_norm > 1e-12]
    if _p_nz.numel() > 0:
        _H_r = -(_p_nz * _p_nz.log()).sum().item()
    else:
        _H_r = 0.0
    SD_r = float(np.exp(_H_r)) / max(p_eff, 1)
    SD_r = min(max(SD_r, 1e-12), 1.0)

    _eps_sched = 1e-8
    _log_inv_seo = float(np.log(1.0 / SEO_r))
    _log_inv_sd = float(np.log(1.0 / SD_r))
    _ratio = _log_inv_seo / (_log_inv_sd + _eps_sched)
    alpha_auto = float(1.0 / (1.0 + np.exp(-_ratio)))
    alpha_auto = max(0.0, min(1.0, alpha_auto))
    if fixed_alpha:
        alpha = alpha_fallback
    else:
        alpha = alpha_auto

    q = min(beta_rank, min(d_out, d_in))
    try:
        U_r, S_r, Vh_r = torch.linalg.svd(tau_r, full_matrices=False)
        tau_r_trunc = (U_r[:, :q] * S_r[:q].unsqueeze(0)) @ Vh_r[:q, :]
        norm_r_sq = float((S_r[:q] ** 2).sum().item())

        U_d, S_d, Vh_d = torch.linalg.svd(tau_d, full_matrices=False)
        tau_d_trunc = (U_d[:, :q] * S_d[:q].unsqueeze(0)) @ Vh_d[:q, :]
        norm_d_sq = float((S_d[:q] ** 2).sum().item())

        cross = tau_r_trunc.T @ tau_d_trunc
        cross_norm_sq = float((cross ** 2).sum().item())
        _denom_beta = norm_r_sq * norm_d_sq + 1e-30

        _tsv_raw = cross_norm_sq / _denom_beta
        beta_auto = float(1.0 / (1.0 + np.exp(-_tsv_raw)))
    except Exception:
        beta_auto = 0.5
        _tsv_raw = 0.0
        cross_norm_sq = 0.0
        norm_r_sq = 0.0
        norm_d_sq = 0.0
    if fixed_beta:
        beta = beta_fallback
    else:
        beta = beta_auto

    alpha = max(1e-6, min(1.0 - 1e-6, alpha))
    D_alpha = torch.pow(sigma, alpha) * (rho ** (1.0 - alpha))
    sum_D_sq = (D_alpha ** 2).sum().clamp(min=1e-30).item()
    c_alpha = norm_Yr / (sum_D_sq ** 0.5)

    src_corr = W_r @ G_mod
    sigma_max = sigma.max().clamp(min=1e-30)
    sigma_tol = 1e-8 * sigma_max
    active = sigma > sigma_tol
    sigma_safe = torch.where(active, sigma, torch.ones_like(sigma))
    Lambda_diag = torch.pow(sigma_safe, alpha - 1.0) * (rho ** (1.0 - alpha))
    Lambda_diag = torch.where(active, Lambda_diag, torch.zeros_like(Lambda_diag))
    Vt_src = V_text.T @ src_corr
    core = U_0 @ (Lambda_diag.unsqueeze(1) * (U_0.T @ Vt_src))
    T_Xs_T = c_alpha * (V_text @ core)

    theta_G_mod = W_base @ G_mod
    consensus_tau = beta * tau_r + (1.0 - beta) * tau_d
    consensus_term = float(eta) * (consensus_tau @ G_text)

    A = G_mod + float(eta) * G_text
    B_mat = T_Xs_T - theta_G_mod + consensus_term

    I_in = torch.eye(d_in, dtype=G_mod.dtype, device=device)
    max_diag = A.diag().abs().max().clamp(min=1e-20)
    A_reg = A + (eps_solve * max_diag) * I_in
    tau_m = torch.linalg.solve(A_reg, B_mat.T).T
    W_merged = W_base + tau_m

    Y_m = W_merged @ G_mod @ W_merged.T
    Y_t_m = W_merged @ G_text @ W_merged.T
    Y_m = 0.5 * (Y_m + Y_m.T)
    Y_t_m = 0.5 * (Y_t_m + Y_t_m.T)
    Y_r_out = G_Yr
    Y_t_r = YtYt

    Y_r_Ptext_energy = float(sigma_sq.sum().clamp(min=0).item())
    Y_r_energy = norm_Yr_sq
    Y_m_energy = torch.trace(Y_m).clamp(min=0).item()
    Y_m_Ptext = V_text.T @ Y_m @ V_text
    Y_m_Ptext_energy = torch.trace(Y_m_Ptext).clamp(min=0).item()

    SEO_before = Y_r_Ptext_energy / Y_r_energy if Y_r_energy > 1e-12 else 0.0
    SEO_after = Y_m_Ptext_energy / Y_m_energy if Y_m_energy > 1e-12 else 0.0

    def _sd_uniformity(Y_sub_pp, p_eff):
        e = torch.linalg.eigvalsh(Y_sub_pp).clamp(min=0)
        tot = e.sum().clamp(min=1e-30)
        p = e / tot
        p_nz = p[p > 1e-12]
        entropy = -(p_nz * p_nz.log()).sum().item()
        if p_eff <= 1:
            return 1.0, entropy, float(np.exp(entropy))
        sd = entropy / float(np.log(p_eff))
        eff_rank = float(np.exp(entropy))
        return sd, entropy, eff_rank

    Y_r_tt = V_text.T @ Y_r_out @ V_text
    Y_m_tt = V_text.T @ Y_m @ V_text
    SD_before, SD_entropy_before, SD_effrank_before = _sd_uniformity(Y_r_tt, p_eff)
    SD_after,  SD_entropy_after,  SD_effrank_after  = _sd_uniformity(Y_m_tt, p_eff)

    Yr_norm = Y_r_energy ** 0.5
    Ym_norm = Y_m_energy ** 0.5
    dY_r = (Y_m - Y_r_out).norm().item()
    Yr_rel_change = dY_r / (Yr_norm + 1e-20)

    Yts_energy = torch.trace(Y_t_r).clamp(min=0).item()
    Ytm_energy = torch.trace(Y_t_m).clamp(min=0).item()
    Yts_norm = Yts_energy ** 0.5
    Ytm_norm = Ytm_energy ** 0.5
    dY_t = (Y_t_m - Y_t_r).norm().item()
    Yt_rel_change = dY_t / (Yts_norm + 1e-20)

    _DD = (D_alpha ** 2).detach()
    _DD_tot = _DD.sum().clamp(min=1e-30)
    _DD_p = (_DD / _DD_tot)
    _DD_nz = _DD_p[_DD_p > 1e-12]
    _DD_entropy = -(_DD_nz * _DD_nz.log()).sum().item()
    _SD_target = _DD_entropy / float(np.log(p_eff)) if p_eff > 1 else 1.0

    diagnostics = {
        'method': 'dcat_closed_form',
        'eta': float(eta),
        'top_p': p_eff,
        'alpha': alpha,
        'beta': beta,
        'alpha_auto': alpha_auto,
        'beta_auto': beta_auto,
        'SEO_r': SEO_r,
        'SD_r': SD_r,
        'alpha_ratio_raw': _ratio,
        'tsv_raw': _tsv_raw,
        'tsv_interference': beta_auto,
        'c_alpha': c_alpha,
        'SD_target': _SD_target,
        'norm_tau_r': tau_r.norm().item(),
        'norm_tau_m': tau_m.norm().item(),
        'norm_Yr': norm_Yr, 'rho': rho,
        'SEO_before': SEO_before,
        'SEO_after': SEO_after,
        'SEO_delta': SEO_after - SEO_before,
        'SD_before': SD_before,
        'SD_after': SD_after,
        'SD_delta': SD_after - SD_before,
        'SD_entropy_before': SD_entropy_before,
        'SD_entropy_after': SD_entropy_after,
        'SD_effrank_before': SD_effrank_before,
        'SD_effrank_after': SD_effrank_after,
        'Yr_norm_before': Yr_norm,
        'Yr_norm_after': Ym_norm,
        'Yr_rel_change': Yr_rel_change,
        'Ym_over_Yr': Ym_norm / norm_Yr if norm_Yr > 1e-12 else 0.0,
        'Yt_norm_before': Yts_norm,
        'Yt_norm_after': Ytm_norm,
        'Yt_rel_change': Yt_rel_change,
        'weight_delta_norm': (W_merged - W_r).norm().item(),
        'weight_orig_norm': W_r.norm().item(),
        'weight_relative_change': (
            (W_merged - W_r).norm().item() / (W_r.norm().item() + 1e-20)),
    }
    return W_merged.cpu(), diagnostics


def string_match(answer, prediction, choices):
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))
    prediction, answer = prediction.strip(), answer.strip()
    if isinstance(choices, str):
        choices = json.loads(choices)
    labels = "ABCDEFGHIJ"
    letter_to_idx = {labels[i]: i for i in range(min(len(labels), len(choices)))}
    answer_idx = None
    for i, c in enumerate(choices):
        c_text = re.sub(r'^\([A-Ea-e]\)\s*', '', c).strip()
        a_text = re.sub(r'^\([A-Ea-e]\)\s*', '', answer).strip()
        if c_text.lower() == a_text.lower():
            answer_idx = i; break
    pred_letter = prediction.upper() if re.match(r'^[A-Ea-e]$', prediction) else None
    if pred_letter and pred_letter in letter_to_idx and answer_idx is not None:
        return letter_to_idx[pred_letter] == answer_idx
    for i, choice in enumerate(choices):
        ct = re.sub(r'^\([A-Ea-e]\)\s*', '', choice).strip()
        if prediction.lower() == ct.lower():
            return i == answer_idx if answer_idx is not None else prediction.lower() == answer.lower()
    p_tok = tokenize(prediction)
    a_tok = tokenize(answer)
    if not p_tok:
        return False
    incorrect = set()
    for c in choices:
        ct = tokenize(c)
        if ct != a_tok:
            incorrect.update(ct - a_tok)
    return a_tok.issubset(p_tok) and p_tok.isdisjoint(incorrect)


def evaluate_mmau_predictions(results_data):
    corr, total = 0, 0
    task_m = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_m = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    for s in results_data:
        pred = s.get('model_output', '')
        ans = s.get('answer', '')
        task = s.get('task', '')
        diff = s.get('difficulty', '')
        choices = s.get('choices', [])
        if not ans or not choices:
            continue
        if string_match(ans, pred, choices):
            corr += 1
            if task in task_m: task_m[task][0] += 1
            if diff in diff_m: diff_m[diff][0] += 1
        total += 1
        if task in task_m: task_m[task][1] += 1
        if diff in diff_m: diff_m[diff][1] += 1
    out = {"total_accuracy": (corr / total * 100) if total > 0 else 0,
           "total_correct": corr, "total_samples": total}
    for k, v in task_m.items():
        out[f"task_{k}"] = (v[0] / v[1] * 100) if v[1] > 0 else 0
    for k, v in diff_m.items():
        out[f"diff_{k}"] = (v[0] / v[1] * 100) if v[1] > 0 else 0
    return out


def extract_answer_letter(s):
    s = s.strip()
    for p in ["The best answer is", "The correct answer is", "The answer is",
              "The answer", "The best option is", "The correct option is",
              "Best answer:", "Best option:", "Answer:", "Option:"]:
        s = s.replace(p, "")
    s = s.strip()
    if re.match(r'^[A-D][\.\)\s,:]', s): return s[0]
    if re.match(r'^[A-D]$', s): return s[0]
    m = re.search(r'\b([A-D])\b', s)
    if m: return m.group(1)
    m = re.search(r'[A-D]', s)
    if m: return m.group(0)
    return ""


def evaluate_videomme_predictions(preds):
    dur_m = {"short": [0, 0], "medium": [0, 0], "long": [0, 0]}
    res = {"total_correct": 0, "total_samples": 0}
    for p in preds:
        gt = p.get("answer", "").strip()
        pl = p.get("pred_answer", "")
        d = p.get("duration", "short")
        res["total_samples"] += 1
        if d in dur_m: dur_m[d][1] += 1
        if pl == gt:
            res["total_correct"] += 1
            if d in dur_m: dur_m[d][0] += 1
    tot = res["total_samples"]
    res["total_accuracy"] = (res["total_correct"] / tot * 100) if tot > 0 else 0
    for d, (c, t) in dur_m.items():
        res[f"dur_{d}_accuracy"] = (c / t * 100) if t > 0 else 0
        res[f"dur_{d}_correct"] = c
        res[f"dur_{d}_total"] = t
    return res


def format_videomme_prompt(sample):
    q = sample.get("question", "")
    opts = sample.get("options", [])
    opts_text = "\n".join(opts)
    return f"{q}\n{opts_text}\n\nAnswer with the option letter (A, B, C, or D)."


def run_video_inference(model_dict, video_path, prompt, max_new_tokens=64):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
    from decord import VideoReader, cpu as decord_cpu
    from PIL import Image as PILImage

    tokenizer = model_dict["tokenizer"]; model = model_dict["model"]
    image_processor = model_dict["image_processor"]
    device = next(model.parameters()).device

    vr = VideoReader(video_path, ctx=decord_cpu(0))
    total = len(vr)
    num_frames = 16
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    pil_frames = [PILImage.fromarray(f) for f in frames]

    images_tensor = image_processor.preprocess(
        pil_frames, return_tensors="pt"
    )["pixel_values"].to(device=device, dtype=torch.bfloat16)

    full_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        out_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=[images_tensor],
            image_sizes=[pil_frames[0].size],
            modalities=["video"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    response = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    if prompt in response:
        response = response[response.index(prompt) + len(prompt):].strip()
    return response


def compute_proj_energy_div_mixed(H_text, H_mod, k=32, alpha=0.5, beta=0.5, device=None):
    if device is None:
        device = torch.device('cuda:0')
    H_text = H_text.float().to(device)
    H_mod = H_mod.float().to(device)
    _, _, Vt = torch.linalg.svd(H_text, full_matrices=False)
    k_eff = min(k, Vt.shape[0])
    U_k = Vt[:k_eff, :].T
    proj = H_mod @ U_k
    proj_energy = (proj ** 2).sum(dim=1)
    mod_energy = (H_mod ** 2).sum(dim=1)
    proj_ratio = (proj_energy / (mod_energy + 1e-10)).mean().item()

    _, s_mod, _ = torch.linalg.svd(proj, full_matrices=False)
    svals_sq = (s_mod ** 2).clamp(min=0)
    p = svals_sq / (svals_sq.sum() + 1e-20)
    p_nz = p[p > 1e-12]
    if p_nz.numel() == 0:
        eff_rank = 1.0
    else:
        entropy = -(p_nz * p_nz.log()).sum()
        eff_rank = torch.exp(entropy).item()
    eff_rank_ratio = eff_rank / max(k_eff, 1)

    nuc = svals_sq.sqrt().sum().item()
    fro = (svals_sq.sum().clamp(min=1e-20).sqrt()).item()
    uniformity = nuc / (fro * (k_eff ** 0.5) + 1e-20)

    val = proj_ratio * (eff_rank_ratio ** alpha) * (uniformity ** beta)
    return {
        'proj_energy_div_mixed': val,
        'proj_ratio': proj_ratio,
        'eff_rank': eff_rank,
        'eff_rank_ratio': eff_rank_ratio,
        'uniformity': uniformity,
        'k': k_eff,
    }


def measure_alignment_layer21(model, tokenizer, pipe, data,
                              modality='audio', layer=21, k=32):
    import librosa

    device = next(model.parameters()).device
    mod_tokens, text_tokens = [], []
    processor = pipe.processor

    for item in tqdm(data, desc=f"  Align L{layer}", leave=False):
        try:
            raw = item.get('modal_inputs', {})
            if 'audio' in raw:
                paths = raw['audio']
                audio_path = paths[0] if isinstance(paths, list) else paths
            elif 'audio_path' in item:
                audio_path = item['audio_path']
            else:
                continue

            audio, _ = librosa.load(audio_path, sr=16000)
            conv_data = item.get('conversations', None)
            if conv_data is not None:
                prompt_text = conv_data[0]['value']
            else:
                prompt_text = f"<|audio|>\n{format_mmau_prompt(item)}"

            inputs = processor(text=prompt_text, audio=audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outs = pipe.model(**inputs, output_hidden_states=True, return_dict=True)

            hs = outs.hidden_states[layer][0].cpu().float()

            if "audio_token_start_idx" not in inputs or "audio_token_len" not in inputs:
                continue

            fip = inputs["audio_token_start_idx"][0].item()
            num_audio_tokens = inputs["audio_token_len"][0].item()

            if num_audio_tokens <= 0:
                continue

            m_tok = hs[fip : fip + num_audio_tokens]
            t_tok = torch.cat([hs[:fip], hs[fip + num_audio_tokens:]], dim=0)

            if m_tok.shape[0] > 0:
                mod_tokens.append(m_tok)
            if t_tok.shape[0] > 0:
                if t_tok.shape[0] > 30:
                    g = torch.Generator().manual_seed(42 + layer)
                    idx = torch.randperm(t_tok.shape[0], generator=g)[:30]
                    t_tok = t_tok[idx]
                text_tokens.append(t_tok)
        except Exception:
            continue

    if not mod_tokens or not text_tokens:
        return {'proj_energy_div_mixed': 0.0, 'num_samples': 0}
    H_mod = torch.cat(mod_tokens, dim=0)
    H_text = torch.cat(text_tokens, dim=0)
    result = compute_proj_energy_div_mixed(H_text, H_mod, k=k)
    result['num_mod_tokens'] = int(H_mod.shape[0])
    result['num_text_tokens'] = int(H_text.shape[0])
    return result


def run_audio_inference(model_dict, audio_path, prompt, max_new_tokens=128):
    import librosa
    pipe = model_dict["image_processor"]
    device = next(model_dict["model"].parameters()).device
    try:
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = pipe.processor(text=prompt, audio=audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = pipe.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
        return pipe.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Inference error: {e}")
        return ""

def run_full_evaluation(model, tokenizer, image_processor, eval_data,
                        tag="before", eval_mode='mmau',
                        num_align_samples=64, align_modality='audio',
                        skip_accuracy=False, verbose=False):
    model_dict = {"tokenizer": tokenizer, "model": model,
                  "image_processor": image_processor}
    print(f"\n{'='*60}\n  [{tag.upper()}] Evaluation ({eval_mode})\n{'='*60}")

    if num_align_samples > 0:
        align_data = eval_data[:min(num_align_samples, len(eval_data))]
    else:
        align_data = eval_data
    align = measure_alignment_layer21(
        model, tokenizer, image_processor, align_data,
        modality=align_modality, layer=21, k=32)
    print(f"  [{tag}] proj_energy_div_mixed(L21,k=32): "
          f"{align.get('proj_energy_div_mixed', 0.0):.4f} "
          f"(proj_ratio={align.get('proj_ratio', 0.0):.4f}, "
          f"eff_rank={align.get('eff_rank', 0.0):.2f}, "
          f"uniformity={align.get('uniformity', 0.0):.4f}, "
          f"n_mod={align.get('num_mod_tokens', 0)})")

    if skip_accuracy:
        print(f"  [{tag}] SKIPPING accuracy evaluation (skip_accuracy=True)")
        acc = {"total_accuracy": float('nan'), "total_correct": 0,
               "total_samples": 0, "skipped": True}
        return {"accuracy": acc, "alignment_layer21_k32": align}

    results_data = []
    errors = 0
    running_correct = 0
    if eval_mode == 'videomme':
        raise NotImplementedError("VideoMME evaluation not supported in audio pipeline.")
    else:
        for si, sample in enumerate(eval_data):
            ap = sample.get("audio_path", "")
            prompt = f"<|audio|>\n{format_mmau_prompt(sample)}"
            try:
                output = run_audio_inference(model_dict, ap, prompt)
            except Exception as e:
                output = f"ERROR: {e}"; errors += 1
            gt_answer = sample.get("answer", "")
            choices = sample.get("choices", [])
            is_correct = string_match(gt_answer, output, choices) if gt_answer and choices else False
            if is_correct:
                running_correct += 1
            running_acc = (running_correct / (si + 1)) * 100

            if verbose:
                q_snippet = sample.get("question", sample.get("instruction", ""))[:80]
                mark = "✓" if is_correct else "✗"
                print(f"  [{tag}] {si+1}/{len(eval_data)} {mark}  "
                      f"acc={running_acc:.1f}%  "
                      f"GT=\"{gt_answer[:50]}\"  "
                      f"pred=\"{output[:60]}\"  "
                      f"task={sample.get('task','')}  "
                      f"diff={sample.get('difficulty','')}")
            elif si < 3:
                logging.info(f"    [DEBUG sample {si}] output={repr(output[:200])}")
            elif (si + 1) % 100 == 0:
                print(f"  [{tag}] {si+1}/{len(eval_data)}  running_acc={running_acc:.1f}%  errors={errors}")

            results_data.append({
                "audio_path": ap,
                "task": sample.get("task", ""),
                "difficulty": sample.get("difficulty", ""),
                "choices": sample.get("choices", []),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "model_output": output,
            })
        acc = evaluate_mmau_predictions(results_data)

    print(f"  [{tag}] Accuracy: {acc.get('total_accuracy', 0):.2f}% "
          f"({acc.get('total_correct', 0)}/{acc.get('total_samples', 0)}) errors={errors}")
    return {
        "accuracy": acc,
        "alignment_layer21_k32": align,
        "predictions": results_data,
    }


def main():
    parser = argparse.ArgumentParser(
        description='DCAT Closed-Form Progressive Merge + Before/After Eval')
    parser.add_argument('--recipient-model-path', '--source-model-path',
                        dest='recipient_model_path', required=True,
                        help='θ₀ + τ_r (recipient model to be improved)')
    parser.add_argument('--donor-model-paths', required=True,
                        help='Comma-separated donor checkpoint paths')
    parser.add_argument('--model-base', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--result-dir', default='results_iter1')
    parser.add_argument('--config-tag', default='')
    parser.add_argument('--exp-name', default='',
                        help='Experiment name for CSV grouping. '
                             'When set, results go to experiment_results_{exp_name}.csv')
    parser.add_argument('--start-layer', type=int, default=0)
    parser.add_argument('--end-layer', type=int, default=31)
    parser.add_argument('--skip-layers', default='')
    parser.add_argument('--proj-groups', default='mlp_full',
                        help='attn_full, mlp_full, or all')
    parser.add_argument('--eta', '--gamma', type=float, default=1e5,
                        dest='eta',
                        help='η: text regularization weight (paper Eq. 13). '
                             'Controls alignment-vs-text balance in '
                             'A = G_mod + η·G_text. Typical range {1e4–1e7}.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='α ∈ (0,1). Fallback fixed-α for target T(α). '
                             'Overridden by auto-schedule (paper Eq. 14) '
                             'using SEO_r and SD_r.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='β ∈ [0,1]. Fixed β or fallback for recipient/donor '
                             'weighting in L_text.')
    parser.add_argument('--fixed-alpha', action='store_true',
                        help='Use fixed --alpha value instead of auto-scheduling '
                             'via SEO/SD (paper Eq. 14).')
    parser.add_argument('--fixed-beta', action='store_true',
                        help='Use fixed --beta value instead of auto-scheduling '
                             'via TSV interference (paper Eq. 16).')
    parser.add_argument('--top-p', '-k', type=int, default=256,
                        dest='top_p',
                        help='k: text subspace dimension (paper §3.1)')
    parser.add_argument('--beta-rank', type=int, default=None,
                        help='q: rank for TSV truncated SVD in β schedule '
                             '(paper Eq. 16). Default = top_p.')
    parser.add_argument('--eps-solve', type=float, default=1e-6,
                        help='Regularization for SPD linear solve')
    parser.add_argument('--num-samples', type=int, default=128,
                        help='Calibration samples')
    parser.add_argument('--num-align-samples', type=int, default=64,
                        help='Samples for alignment measurement')
    parser.add_argument('--lora-alpha', type=int, default=256)
    parser.add_argument('--lora-r', type=int, default=128)
    parser.add_argument('--eval-mode', choices=['mmau', 'videomme'], default='mmau')
    parser.add_argument('--num-eval-samples', type=int, default=0,
                        help='0 = use full set (MMAU full test-mini)')
    parser.add_argument('--videomme-num-eval', type=int, default=900)
    parser.add_argument('--videomme-seed', type=int, default=77)
    parser.add_argument('--calib-modality', choices=['audio', 'video'], default='audio')
    parser.add_argument('--calib-dataset', choices=['mmau', 'avqa', 'videomme'], default='mmau')
    parser.add_argument('--mmau-data-root', default=DATA_DIR_MMAU)
    parser.add_argument('--mmau-testset', default='full')
    parser.add_argument('--skip-before', default='',
                        help='Cached BEFORE result JSON to skip before eval')
    parser.add_argument('--skip-after', action='store_true', default=False,
                        help='Skip the AFTER evaluation entirely. Useful for sweeps '
                             'where evaluation is run separately on the saved model.')
    parser.add_argument('--skip-before-accuracy', action='store_true', default=False,
                        help='Skip the BEFORE accuracy evaluation entirely '
                             '(alignment is still measured).')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print per-sample inference results during evaluation '
                             '(GT, prediction, match, running accuracy).')
    args = parser.parse_args()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    result_dir = args.result_dir if os.path.isabs(args.result_dir) else os.path.join(SCRIPT_DIR, args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, 'dcat_merge_log.txt')
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])

    donor_paths = [p.strip() for p in args.donor_model_paths.split(',') if p.strip()]
    K = len(donor_paths)
    proj_groups_ordered, proj_paths_all = resolve_proj_paths(args.proj_groups)
    skip_layers = set(int(x) for x in args.skip_layers.split(',') if x.strip())

    logging.info("=" * 70)
    logging.info("  DCAT Closed-Form Progressive Merge + Eval")
    logging.info("=" * 70)
    logging.info(f"  Recipient:    {args.recipient_model_path}")
    logging.info(f"  Donors (K={K}):")
    for i, dp in enumerate(donor_paths):
        logging.info(f"    [{i}] {dp}")
    logging.info(f"  Base LLM:     {args.model_base}")
    logging.info(f"  Output:       {args.output_dir}")
    logging.info(f"  Layers:       {args.start_layer}–{args.end_layer}"
                 + (f"  (skip: {sorted(skip_layers)})" if skip_layers else ""))
    logging.info(f"  Proj groups:  {proj_groups_ordered}  -> paths: {proj_paths_all}")
    logging.info(f"  η = {args.eta},  α(fallback) = {args.alpha},  "
                 f"β(fallback) = {args.beta},  k = {args.top_p},  "
                 f"eps_solve = {args.eps_solve}")
    logging.info(f"  α/β auto-scheduled per projection (paper Eq. 14/16)")
    logging.info(f"  Samples:      calib={args.num_samples}, align={args.num_align_samples}, "
                 f"eval={args.num_eval_samples if args.eval_mode=='mmau' else args.videomme_num_eval}")
    logging.info(f"  Eval mode:    {args.eval_mode}")
    logging.info("=" * 70)


    logging.info(f"\n[Step 1] Loading calibration data (dataset={args.calib_dataset}, "
                 f"modality={args.calib_modality})...")
    if args.calib_dataset == 'mmau':
        calibration_data = load_mmau_audio_data(
            args.mmau_data_root, testset=args.mmau_testset,
            num_samples=args.num_samples)
    elif args.calib_dataset == 'videomme':
        calibration_data = load_videomme_eval_data(
            num_samples=args.num_samples, seed=42)
        for item in calibration_data:
            vp = item.get('video_path', '')
            if 'modal_inputs' not in item:
                item['modal_inputs'] = {'video': [vp]}
            if 'conversations' not in item:
                item['conversations'] = [{'value': f"<image>\n{format_videomme_prompt(item)}"}]
    else:
        if args.calib_modality == 'video':
            calibration_data = load_avqa_video_data(args.num_samples)
        else:
            calibration_data = load_avqa_audio_data(AVQA_DATA_ROOT, args.num_samples)
    logging.info(f"  Calibration: {len(calibration_data)} samples")

    if args.eval_mode == 'videomme':
        _vme_n = args.num_eval_samples if args.num_eval_samples > 0 else 0
        eval_data = load_videomme_eval_data(_vme_n,
                                            seed=args.videomme_seed)
        logging.info(f"  VideoMME eval: {len(eval_data)} samples "
                     f"(seed={args.videomme_seed})")
    else:
        if args.num_eval_samples > 0:
            variant_key = "test_mini"
            input_json = os.path.join(args.mmau_data_root, f"{variant_key}.json")
            with open(input_json) as f:
                eval_raw = json.load(f)
            for it in eval_raw:
                if 'audio_path' in it and not os.path.isabs(it['audio_path']):
                    it['audio_path'] = os.path.join(args.mmau_data_root, it['audio_path'])
            eval_raw = [it for it in eval_raw if os.path.exists(it.get('audio_path', ''))]
            import random
            rng = random.Random(42)
            idx = list(range(len(eval_raw))); rng.shuffle(idx)
            eval_data = [eval_raw[i] for i in idx[:args.num_eval_samples]]
        else:
            eval_data = calibration_data

    logging.info("\n[Step 2] Loading recipient model...")
    tokenizer, model, image_processor = load_model(args.recipient_model_path, args.model_base)

    donor_names = [os.path.basename(dp)[:20] for dp in donor_paths]
    if args.config_tag:
        config_name = args.config_tag
    else:
        layer_tag = f"L{args.start_layer}-{args.end_layer}"
        pg_tag = '+'.join(proj_groups_ordered)
        config_name = (f"DCAT_K{K}_{'_'.join(donor_names)}"
                       f"_eta{args.eta}_k{args.top_p}"
                       f"_{layer_tag}_{pg_tag}_s{args.num_samples}")

    before_path = os.path.join(result_dir, f"{config_name}_before.json")
    if args.skip_before and os.path.isfile(args.skip_before):
        logging.info(f"\n[BEFORE] SKIPPED — using cached {args.skip_before}")
        with open(args.skip_before) as f:
            before_result = json.load(f)
        logging.info(f"  Cached BEFORE Acc: "
                     f"{before_result['accuracy']['total_accuracy']:.2f}%")
    else:
        align_mod = args.calib_modality if args.eval_mode != 'videomme' else 'video'
        before_result = run_full_evaluation(
            model, tokenizer, image_processor, eval_data,
            tag="before", eval_mode=args.eval_mode,
            num_align_samples=args.num_align_samples, align_modality=align_mod,
            skip_accuracy=args.skip_before_accuracy,
            verbose=args.verbose)
        with open(before_path, "w") as f:
            json.dump(before_result, f, indent=2)
        logging.info(f"  Saved BEFORE: {before_path}")

    logging.info("\n" + "=" * 70)
    logging.info(f"  DCAT Progressive Merge  (K={K}, η={args.eta}, k={args.top_p})")
    logging.info("=" * 70)

    samples = extract_initial_hidden_states(
        model, tokenizer, image_processor, calibration_data,
        args.start_layer, modality=args.calib_modality)
    if not samples:
        logging.error("ERROR: no calibration samples captured. Aborting."); return
    logging.info(f"  Captured {len(samples)} calibration samples")
    total_mod = sum(s['H_src'].shape[0] for s in samples)
    total_txt = sum(s['H_text'].shape[0] for s in samples)
    logging.info(f"  Total modality tokens: {total_mod}, text tokens: {total_txt}")

    logging.info("  Loading base LLM weights...")
    base_weights = load_base_model_weights(args.model_base)
    logging.info(f"  Loading {K} donor model weights...")
    donor_adapters = [load_donor_weights(dp) for dp in donor_paths]

    merge_stats = {}
    t_total = time.time()

    for layer_idx in range(args.start_layer, args.end_layer + 1):
        if layer_idx in skip_layers:
            logging.info(f"\n  [SKIP] layer {layer_idx}")
            continue
        logging.info(f"\n{'─'*60}\n  Layer {layer_idx}/{args.end_layer}\n{'─'*60}")
        t_layer = time.time()

        dep_groups = split_into_dependency_groups(proj_paths_all)

        for gi, group_paths in enumerate(dep_groups):
            group_label = ','.join(p.split('.')[-1] for p in group_paths)
            logging.info(f"    ── dep-group {gi+1}/{len(dep_groups)}: [{group_label}] ──")

            t_stack = time.time()
            proj_feats = stack_features_for_layer(model, layer_idx, group_paths,
                                                  samples, device,
                                                  capture_residuals=False)
            logging.info(f"    Stacked features ({group_label}) in {time.time()-t_stack:.1f}s")

            for proj_path in group_paths:
                feats = proj_feats[proj_path]
                G_mod = feats['G_s']; G_text = feats['G_t']
                if G_mod is None or G_text is None:
                    logging.warning(f"    {proj_path}: no features, skip")
                    continue
                W_r = get_current_effective_weight(model, layer_idx, proj_path)
                donor_Ws = [get_effective_weight(base_weights, da, layer_idx, proj_path,
                                                 args.lora_alpha, args.lora_r)
                            for da in donor_adapters]
                base_key = f'model.layers.{layer_idx}.{proj_path}.weight'
                W_base_l = base_weights[base_key].float()

                t0 = time.time()
                W_merged, diag = compute_tau_m_mi_surrogate(
                    W_base_l, W_r, donor_Ws, G_mod, G_text,
                    eta=args.eta, top_p=args.top_p,
                    device=device, eps_solve=args.eps_solve,
                    alpha=args.alpha, beta=args.beta,
                    fixed_alpha=args.fixed_alpha,
                    beta_rank=args.beta_rank,
                    fixed_beta=args.fixed_beta)
                dt = time.time() - t0
                diag['compute_time'] = dt
                stat_key = f"layer{layer_idx}.{proj_path.split('.')[-1]}"
                merge_stats[stat_key] = diag

                logging.info(
                    f"    {proj_path.split('.')[-1]}: "
                    f"α={diag['alpha']:.4f} (SEO_r={diag['SEO_r']:.4f}, SD_r={diag['SD_r']:.4f}), "
                    f"β={diag['beta']:.3f} (TSV={diag['tsv_interference']:.3f}), "
                    f"η={diag['eta']:.1e}, ‖τ_m‖={diag['norm_tau_m']:.3f}, "
                    f"ρ={diag['rho']:.3f}  ({dt:.1f}s)")
                logging.info(
                    f"      SEO: {diag['SEO_before']:.4f} -> {diag['SEO_after']:.4f}  "
                    f"(Δ={diag['SEO_delta']:+.4f})    "
                    f"SD:  {diag['SD_before']:.4f} -> {diag['SD_after']:.4f}  "
                    f"(Δ={diag['SD_delta']:+.4f})")
                logging.info(
                    f"      SD eff_rank(in P_text, k={diag['top_p']}): "
                    f"{diag['SD_effrank_before']:.2f} -> {diag['SD_effrank_after']:.2f}   "
                    f"entropy: {diag['SD_entropy_before']:.3f} -> "
                    f"{diag['SD_entropy_after']:.3f}")
                logging.info(
                    f"      Y_r norm: {diag['Yr_norm_before']:.3e} -> {diag['Yr_norm_after']:.3e}   "
                    f"rel ΔY_r = {diag['Yr_rel_change']:.4f}")
                logging.info(
                    f"      Y_txt norm: {diag['Yt_norm_before']:.3e} -> {diag['Yt_norm_after']:.3e}   "
                    f"rel ΔY_txt = {diag['Yt_rel_change']:.4f}")
                logging.info(
                    f"      ΔW/W = {diag['weight_relative_change']:.4f}  "
                    f"(‖W‖={diag['weight_orig_norm']:.3e}, "
                    f"‖ΔW‖={diag['weight_delta_norm']:.3e})    "
                    f"‖Y_m‖/‖Y_r‖={diag['Ym_over_Yr']:.4f}")

                update_model_weight_inplace(model, layer_idx, proj_path, W_merged)
                del W_r, W_merged, donor_Ws
        t_prop = time.time()
        propagate_through_layer(model, layer_idx, samples, device)
        logging.info(f"    propagated in {time.time()-t_prop:.1f}s,  "
                     f"layer total {time.time()-t_layer:.1f}s")

    dt_total = time.time() - t_total
    logging.info(f"\n  Merge complete: {dt_total:.0f}s ({dt_total/60:.1f} min)")

    try:
        del samples
    except NameError:
        pass
    try:
        del base_weights
    except NameError:
        pass
    try:
        del donor_adapters
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    logging.info(f"\n  Saving full merged model to {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_state = model.state_dict()
    from safetensors.torch import save_file as save_safetensors
    shard_size_bytes = 5 * 1024 * 1024 * 1024
    shards = []
    current_shard = OrderedDict()
    current_size = 0
    for k, v in model_state.items():
        v_cpu = v.detach().cpu()
        elem_size = v_cpu.element_size() * v_cpu.nelement()
        if current_size + elem_size > shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = OrderedDict()
            current_size = 0
        current_shard[k] = v_cpu
        current_size += elem_size
    if current_shard:
        shards.append(current_shard)

    weight_map = {}
    for i, shard in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        shard_path = os.path.join(args.output_dir, shard_name)
        save_safetensors(shard, shard_path)
        for k in shard.keys():
            weight_map[k] = shard_name
        logging.info(f"    Saved shard: {shard_name} ({len(shard)} tensors)")

    index = {"metadata": {"total_size": sum(v.nelement() * v.element_size()
                                             for v in model_state.values())},
             "weight_map": weight_map}
    idx_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)
    logging.info(f"  Saved index: {idx_path}")

    import shutil
    for fname in ['config.json', 'generation_config.json',
                  'tokenizer.json', 'tokenizer_config.json',
                  'special_tokens_map.json']:
        src_f = os.path.join(args.recipient_model_path, fname)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(args.output_dir, fname))

    if args.skip_after:
        logging.info("\n[AFTER] SKIPPED (--skip-after)")
        after_result = {"accuracy": {"total_accuracy": float('nan'),
                                     "total_correct": 0,
                                     "total_samples": 0,
                                     "skipped": True},
                        "alignment_layer21_k32": {}}
    else:
        logging.info("\n" + "=" * 70)
        logging.info("  [AFTER] Evaluating merged model...")
        logging.info("=" * 70)
        align_mod = args.calib_modality if args.eval_mode != 'videomme' else 'video'
        after_result = run_full_evaluation(
            model, tokenizer, image_processor, eval_data,
            tag="after", eval_mode=args.eval_mode,
            num_align_samples=args.num_align_samples, align_modality=align_mod,
            verbose=args.verbose)
        after_path = os.path.join(result_dir, f"{config_name}_after.json")
        with open(after_path, "w") as f:
            json.dump(after_result, f, indent=2)
        logging.info(f"  Saved AFTER: {after_path}")

    before_acc = before_result["accuracy"]["total_accuracy"]
    after_acc = after_result["accuracy"]["total_accuracy"]
    import math as _math
    delta = (after_acc - before_acc) if not _math.isnan(before_acc) else float('nan')
    before_align = before_result.get("alignment_layer21_k32", {})
    after_align = after_result.get("alignment_layer21_k32", {})

    comparison = {
        "config": config_name,
        "method": "dcat_closed_form",
        "K": K, "donor_paths": donor_paths,
        "recipient_model": args.recipient_model_path,
        "eval_mode": args.eval_mode,
        "merge_time_seconds": dt_total,
        "before_accuracy": before_acc,
        "after_accuracy": after_acc,
        "delta_accuracy": delta,
        "before_subcategories": {k: v for k, v in before_result["accuracy"].items()
                                 if k.startswith(("task_", "diff_", "dur_"))},
        "after_subcategories": {k: v for k, v in after_result["accuracy"].items()
                                if k.startswith(("task_", "diff_", "dur_"))},
        "alignment_l21_k32_before": before_align,
        "alignment_l21_k32_after": after_align,
        "alignment_l21_k32_delta": (after_align.get('proj_energy_div_mixed', 0) -
                                    before_align.get('proj_energy_div_mixed', 0)),
        "merge_stats_summary": {
            "mean_weight_change": float(np.mean([s['weight_relative_change']
                                                 for s in merge_stats.values()] or [0])),
            "mean_Ym_over_Yr": float(np.mean([s.get('Ym_over_Yr', 0)
                                              for s in merge_stats.values()] or [0])),
        },
        "hyperparameters": {
            "eta": args.eta, "top_p": args.top_p,
            "alpha": args.alpha,
            "beta": args.beta,
            "alpha_auto_scheduled": True,
            "beta_fixed": args.fixed_beta,
            "eps_solve": args.eps_solve,
            "num_samples": args.num_samples,
            "proj_groups": proj_groups_ordered,
            "layers": f"{args.start_layer}-{args.end_layer}",
            "skip_layers": sorted(skip_layers),
        },
    }
    comp_path = os.path.join(result_dir, f"{config_name}_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  DCAT COMPARISON  (K={K}, {args.eval_mode})")
    print("=" * 70)
    print(f"  Config: {config_name}")
    print(f"  Merge time: {dt_total:.0f}s ({dt_total/60:.1f} min)")
    if _math.isnan(before_acc):
        print(f"  Accuracy:  [skipped] -> {after_acc:.2f}%  (Δ=n/a)")
    else:
        print(f"  Accuracy:  {before_acc:.2f}% -> {after_acc:.2f}%  (Δ={delta:+.2f}%)")

    def _fmt_align(label, a):
        return (f"  [{label:6s}] proj_energy_div_mixed={a.get('proj_energy_div_mixed', 0):.4f}  "
                f"proj_ratio={a.get('proj_ratio', 0):.4f}  "
                f"eff_rank={a.get('eff_rank', 0):.2f}  "
                f"uniformity={a.get('uniformity', 0):.4f}  "
                f"n_mod={a.get('num_mod_tokens', 0)}  "
                f"n_txt={a.get('num_text_tokens', 0)}")
    print("  ── Alignment (L21, k=32) ──")
    print(_fmt_align("before", before_align))
    print(_fmt_align("after",  after_align))
    print(f"  [delta ] proj_energy_div_mixed Δ={comparison['alignment_l21_k32_delta']:+.4f}  "
          f"(proj_ratio Δ={after_align.get('proj_ratio', 0) - before_align.get('proj_ratio', 0):+.4f}, "
          f"eff_rank Δ={after_align.get('eff_rank', 0) - before_align.get('eff_rank', 0):+.2f}, "
          f"uniformity Δ={after_align.get('uniformity', 0) - before_align.get('uniformity', 0):+.4f})")
    ms = comparison["merge_stats_summary"]
    print(f"  Mean ΔW/W={ms['mean_weight_change']:.4f}")
    print(f"  Mean ‖Y_m‖/‖Y_r‖={ms['mean_Ym_over_Yr']:.4f}")
    print(f"  Saved: {comp_path}")
    print("=" * 70)

    import csv
    csv_name = f"experiment_results_{args.exp_name}.csv" if args.exp_name else "experiment_results.csv"
    csv_path = os.path.join(SCRIPT_DIR, csv_name)
    write_header = not os.path.isfile(csv_path)
    row = {
        "exp_name": args.exp_name or "-",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_tag": config_name,
        "eta": args.eta,
        "top_p": args.top_p,
        "alpha": args.alpha,
        "beta": args.beta,
        "beta_fixed": args.fixed_beta,
        "layers": f"{args.start_layer}-{args.end_layer}",
        "before_acc": f"{before_acc:.2f}" if not _math.isnan(before_acc) else "skip",
        "after_acc": f"{after_acc:.2f}",
        "delta_acc": f"{delta:+.2f}" if not _math.isnan(delta) else "n/a",
        "align_before": f"{before_align.get('proj_energy_div_mixed', 0):.4f}",
        "align_after": f"{after_align.get('proj_energy_div_mixed', 0):.4f}",
        "align_delta": f"{comparison['alignment_l21_k32_delta']:+.4f}",
        "proj_ratio_before": f"{before_align.get('proj_ratio', 0):.4f}",
        "proj_ratio_after": f"{after_align.get('proj_ratio', 0):.4f}",
        "eff_rank_before": f"{before_align.get('eff_rank', 0):.2f}",
        "eff_rank_after": f"{after_align.get('eff_rank', 0):.2f}",
        "mean_dW_W": f"{ms['mean_weight_change']:.4f}",
        "merge_time_s": f"{dt_total:.0f}",
    }
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logging.info(f"  📊 Appended to {csv_path}")


if __name__ == '__main__':
    main()
