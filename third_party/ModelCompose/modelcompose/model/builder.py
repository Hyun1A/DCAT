# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from modelcompose.model import *
from modelcompose.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", active_modality=None):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'local-lora' in model_name.lower() and model_base is not None:
            print('Loading LocalLLaVA from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = LocalLlavaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            
            saved_weights = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
            saved_weights = {k: v.to(torch.float16) for k, v in saved_weights.items()}
            model.load_state_dict(saved_weights, strict=False)
        elif 'delta' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)
            
            delta_weights = torch.load(os.path.join(model_path, 'delta_weights.bin'), map_location='cpu')
            base_weights = model.state_dict()
            
            print('Merging Delta weights...')
            for k in delta_weights:
                base_weights[k] = base_weights[k].cpu() + delta_weights[k]
            
            model.load_state_dict(base_weights, strict=False)
            print('Model is loaded...')
        elif 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'local-lora' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LocalLlavaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    elif 'multimodal' in model_name.lower():
        import peft
        import logging as _logging
        if model_base is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = MultimodalConfig.from_pretrained(model_path)

            # ── Load base model to CPU (no device_map) ──
            # device_map="auto" causes meta tensors for custom LocalLoraLinear
            # modules, making subsequent load_state_dict a silent no-op.
            _load_kwargs = {
                'low_cpu_mem_usage': False,
                'torch_dtype': torch.bfloat16,
            }
            
            # Suppress "newly initialized" warning from from_pretrained
            # These weights are expected to be overwritten by adapter_model.bin
            _transformers_logger = _logging.getLogger("transformers.modeling_utils")
            _prev_level = _transformers_logger.level
            _transformers_logger.setLevel(_logging.ERROR)
            # breakpoint()
            model = MultimodalLlamaForCausalLM.from_pretrained(
                model_base, config=cfg_pretrained, ignore_mismatched_sizes=True,
                **_load_kwargs
            )
            _transformers_logger.setLevel(_prev_level)
            
            # ── Materialize meta tensors ──
            # from_pretrained with low_cpu_mem_usage=True + ignore_mismatched_sizes=True
            # leaves custom modules (lora_A/B, modal_projectors) as meta tensors.
            # We must materialize them before load_state_dict and .to(device).
            for name, param in list(model.named_parameters()):
                if param.is_meta:
                    real_param = torch.nn.Parameter(
                        torch.empty(param.shape, dtype=param.dtype, device='cpu'),
                        requires_grad=param.requires_grad,
                    )
                    parts = name.split('.')
                    mod = model
                    for part in parts[:-1]:
                        mod = getattr(mod, part)
                    setattr(mod, parts[-1], real_param)
            for name, buf in list(model.named_buffers()):
                if buf.is_meta:
                    parts = name.split('.')
                    mod = model
                    for part in parts[:-1]:
                        mod = getattr(mod, part)
                    setattr(mod, parts[-1], torch.empty(buf.shape, dtype=buf.dtype, device='cpu'))

            # NOTE: Do NOT call reset_lora_parameters here.
            # The adapter file already contains proper LoRA A/B values.

            # ── Load adapter / projector weights ──
            adapter_path = os.path.join(model_path, 'adapter_model.bin')
            if not os.path.exists(adapter_path):
                adapter_path = os.path.join(model_path, 'mm_projector.bin')
            
            adapter_weights = torch.load(adapter_path, map_location='cpu')
            adapter_weights = {k: v.to(torch.bfloat16) for k, v in adapter_weights.items()}

            # [ACTIVE MODALITY] When active_modality is set (e.g. 'video'),
            # remap 'default-{modality}' adapter keys → 'default' keys.
            # This replaces the corrupted task-vector-merged default LoRA
            # with the correct single-modality LoRA for proper generation.
            if active_modality is not None:
                modality_prefix = f'default-{active_modality}'
                remap_keys = {k: k.replace(modality_prefix, 'default')
                              for k in adapter_weights if modality_prefix in k}
                if remap_keys:
                    print(f"[builder] Remapping {len(remap_keys)} '{modality_prefix}' adapter keys → 'default'")
                    for old_key, new_key in remap_keys.items():
                        adapter_weights[new_key] = adapter_weights.pop(old_key)
                else:
                    print(f"[builder] WARNING: active_modality='{active_modality}' but no '{modality_prefix}' keys found in checkpoint")

            # Use assign=True to replace tensors entirely (avoids silent copy failure on meta tensors)
            model.load_state_dict(adapter_weights, strict=False)
            print(f"[ModelComposeVLM] Adapter loaded: {len(adapter_weights)} keys")
            
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                non_lora_trainables = {k: v.to(torch.bfloat16) for k, v in non_lora_trainables.items()}
                model.load_state_dict(non_lora_trainables, strict=False, assign=True)

            print("[ModelComposeVLM] Model loaded successfully")

        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            
            cfg_pretrained = MultimodalConfig.from_pretrained(model_path)
            model = MultimodalLlamaForCausalLM.from_pretrained(
                model_path, config=cfg_pretrained,
                low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
            )

            for name, module in model.named_modules():
                if isinstance(module, peft.tuners.lora.Linear):
                    for adapter_name in module.lora_A.keys():
                        module.reset_lora_parameters(adapter_name, init_lora_weights=True)
            
        for modal in model.get_model().modal_encoders:
            if modal != 'default':
                model.get_model().modal_encoders[modal].load_model()
        model.to(device=device, dtype=torch.bfloat16)
    else:
        # Load language model
        # Check if this is a full-weight merged model (e.g., LLaMA 3.1 task-vector merges)
        # These have model.safetensors but NO adapter_config.json
        _is_full_weight_merge = (
            os.path.isdir(model_path) and
            (os.path.exists(os.path.join(model_path, 'model.safetensors')) or
             os.path.exists(os.path.join(model_path, 'merged_model.bin'))) and
            not os.path.exists(os.path.join(model_path, 'adapter_config.json'))
        )
        if _is_full_weight_merge:
            # Full-weight merged model (e.g., LLaMA 3.1 task-vector merges):
            # These checkpoints contain base LLM weights + merged deltas + modality-specific
            # components (vision connectors, speech generators, etc.) in a single safetensors.
            # We load the base LLaMA model and overlay the merged LLM weights.
            print(f"Detected full-weight merged model at {model_path}")
            
            # Use model_base if provided, otherwise try to load config from model_path
            if model_base is not None and model_base != "":
                base_for_loading = model_base
            else:
                base_for_loading = model_path
            
            tokenizer = AutoTokenizer.from_pretrained(base_for_loading, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                base_for_loading, low_cpu_mem_usage=True, **kwargs
            )
            
            # Load merged weights and overlay onto the base model
            merged_safetensors = os.path.join(model_path, 'model.safetensors')
            merged_bin = os.path.join(model_path, 'merged_model.bin')
            if os.path.exists(merged_safetensors):
                from safetensors.torch import load_file as _load_safetensors
                merged_weights = _load_safetensors(merged_safetensors)
            elif os.path.exists(merged_bin):
                merged_weights = torch.load(merged_bin, map_location='cpu')
            else:
                raise FileNotFoundError(f"No merged weights found in {model_path}")
            
            # Only load keys that exist in the model (ignore modality-specific keys)
            model_keys = set(model.state_dict().keys())
            matched, skipped = 0, 0
            weights_to_load = {}
            for k, v in merged_weights.items():
                if k in model_keys:
                    weights_to_load[k] = v.to(torch.float16)
                    matched += 1
                else:
                    skipped += 1
            
            model.load_state_dict(weights_to_load, strict=False)
            print(f"Full-weight merged model loaded: {matched} keys matched, {skipped} modality-specific keys skipped")
            del merged_weights, weights_to_load
        elif model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    modal_processors = None
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        modal_processors = vision_tower.image_processor
    elif 'multimodal' in model_name.lower():
        modal_processors = model.get_modal_processors()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, modal_processors, context_len
