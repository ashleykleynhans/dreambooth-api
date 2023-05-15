#!/usr/bin/env python3
import json
import requests

URL = 'http://172.30.61.140:7860'

# FIXME: Lots of unknowns, like how to configure a concept
USE_EMA = False
HAS_EMA = False
CACHE_LATENTS = True
GRADIENT_CHECKPOINTING = True
MODEL_NAME = 'PROTOGEN-TEST'
NUM_TRAIN_EPOCHS = 150
SANITY_PROMPT = 'photo of ohwx person by Tomer Hanuka'
LEARNING_RATE_SCHEDULER = 'constant_with_warmup'
LEARNING_RATE = 0.000001
OPTIMIZER = '8bit AdamW'
MIXED_PRECISION = 'fp16'
ATTENTION = 'xformers'
RESOLUTION = 512

SAVE_CKPT_AFTER = True
SAVE_CKPT_CANCEL = False
SAVE_CKPT_DURING = True
SAVE_EMA = False
SAVE_EMBEDDING_EVERY = 25
SAVE_PREVIEW_EVERY = 5
SAVE_SAFETENSORS = True


CONCEPTS = []

PAYLOAD = {
    "adamw_weight_decay": 0.01,
    "adaptation_beta1": 0,
    "adaptation_beta2": 0,
    "adaptation_d0": 1e-8,
    "adaptation_eps": 1e-8,
    "attention": ATTENTION,
    "cache_latents": CACHE_LATENTS,
    "clip_skip": 1,
    "concepts_list": CONCEPTS,
    "concepts_path": "",
    "custom_model_name": "",
    "noise_scheduler": "DDPM",
    "deterministic": False,
    "ema_predict": False,
    "epoch": 0,
    "epoch_pause_frequency": 0,
    "epoch_pause_time": 0,
    "freeze_clip_normalization": True,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": GRADIENT_CHECKPOINTING,
    "gradient_set_to_none": True,
    "graph_smoothing": 50,
    "half_model": False,
    "train_unfrozen": True,
    "has_ema": HAS_EMA,
    "hflip": False,
    "infer_ema": False,
    "initial_revision": 0,
    "learning_rate": LEARNING_RATE,
    "learning_rate_min": 0.000001,
    "lifetime_revision": 0,
    "lora_learning_rate": 0.0001,
    "lora_model_name": "",
    "lora_unet_rank": 4,
    "lora_txt_rank": 4,
    "lora_txt_learning_rate": 0.00005,
    "lora_txt_weight": 1,
    "lora_weight": 1,
    "lr_cycles": 1,
    "lr_factor": 0.5,
    "lr_power": 1,
    "lr_scale_pos": 0.5,
    "lr_scheduler": LEARNING_RATE_SCHEDULER,
    "lr_warmup_steps": 0,
    "max_token_length": 75,
    "mixed_precision": MIXED_PRECISION,
    "model_name": MODEL_NAME,
    "model_dir": "",
    "model_path": "",
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "offset_noise": 0,
    "optimizer": OPTIMIZER,
    "pad_tokens": True,
    "pretrained_model_name_or_path": "",
    "pretrained_vae_name_or_path": "",
    "prior_loss_scale": False,
    "prior_loss_target": 100,
    "prior_loss_weight": 0.75,
    "prior_loss_weight_min": 0.1,
    "resolution": RESOLUTION,
    "revision": 0,
    "sample_batch_size": 1,
    "sanity_prompt": SANITY_PROMPT,
    "sanity_seed": 420420,
    "save_ckpt_after": SAVE_CKPT_AFTER,
    "save_ckpt_cancel": SAVE_CKPT_CANCEL,
    "save_ckpt_during": SAVE_CKPT_DURING,
    "save_ema": SAVE_EMA,
    "save_embedding_every": SAVE_EMBEDDING_EVERY,
    "save_lora_after": True,
    "save_lora_cancel": False,
    "save_lora_during": True,
    "save_lora_for_extra_net": True,
    "save_preview_every": SAVE_PREVIEW_EVERY,
    "save_safetensors": SAVE_SAFETENSORS,
    "save_state_after": False,
    "save_state_cancel": False,
    "save_state_during": False,
    "scheduler": "ddim",
    "shuffle_tags": False,
    "snapshot": "",
    "split_loss": True,
    "src": "",
    "stop_text_encoder": 1,
    "strict_tokens": False,
    "tf32_enable": False,
    "train_batch_size": 1,
    "train_imagic": False,
    "train_unet": True,
    "use_concepts": True,
    "use_ema": USE_EMA,
    "use_lora": False,
    "use_lora_extended": False,
    "use_subdir": False,
    "v2": False
}


def model_config():
    endpoint = f'{URL}/dreambooth/model_config'

    r = requests.post(
        endpoint
    )

    print(r.status_code)
    response_text = r.json()
    print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    model_config()
