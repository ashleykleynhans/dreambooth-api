#!/usr/bin/env python3
import json
import requests


LEARNING_RATE_SCHEDULER_TYPES = {
    'CONSTANT': 'constant',
    'CONSTANT_WITH_WARMUP': 'constant_with_warmup'
}

SCHEDULER_TYPES = {
    'DDIM': 'ddim',
    'DEISMULTISTEP': 'DEISMultistep'
}

URL = 'http://172.30.61.140:7860'
WEBUI_INSTALLATION_PATH = '/home/ubuntu/stable-diffusion-webui'

USE_EMA = True
HAS_EMA = False
CACHE_LATENTS = True
GRADIENT_CHECKPOINTING = True
MODEL_NAME = 'PROTOGEN-TEST'
SOURCE_MODEL_PATH = '/models/ckpt/ProtoGen_X5.8.safetensors'
NUM_TRAIN_EPOCHS = 200
SANITY_PROMPT = 'photo of ohwx person by Tomer Hanuka'
LEARNING_RATE_SCHEDULER = LEARNING_RATE_SCHEDULER_TYPES['CONSTANT']
LEARNING_RATE = 0.000001
TXT_LEARNING_RATE = 0.000001
STOP_TEXT_ENCODER = 0.75
TRAIN_UNFROZEN =  False
#######################################################
# Constant Learning Rate
#######################################################
LEARNING_RATE_CONSTANT_LINEAR_STARTING_FACTOR = 1
LEARNING_RATE_SCALE_POSITION = 1
#######################################################
OPTIMIZER = '8bit AdamW'
MIXED_PRECISION = 'fp16'
ATTENTION = 'xformers'
RESOLUTION = 512

SAVE_CKPT_AFTER = True
SAVE_CKPT_CANCEL = False
SAVE_CKPT_DURING = True
SAVE_EMA = False
SAVE_EMBEDDING_EVERY = 15
SAVE_PREVIEW_EVERY = 5
SAVE_SAFETENSORS = True
SAVE_CHECKPOINT_TO_SUBDIRECTORY = True
SCHEDULER = SCHEDULER_TYPES['DDIM']


#######################################################
# Concept Configuration                               #
#######################################################
INSTANCE_DATA_DIR = '/home/ubuntu/training-512x512'
CLASS_DATA_DIR = '/home/ubuntu/woman-images'
INSTANCE_PROMPT = 'ohwx woman'
CLASS_PROMPT = 'photo of woman'
SAMPLE_IMAGE_PROMPT = 'photo of ohwx woman'
#######################################################
CLASS_IMAGES_PER_INSTANCE_IMAGE = 20
CLASSIFICATION_CFG_SCALE = 7.5
CLASSIFICATION_STEPS = 40
#######################################################
NUMBER_OF_SAMPLES_TO_GENERATE = 1
SAMPLE_SEED = -1
SAMPLE_CFG_SCALE = 7.5
SAMPLE_STEPS = 20
#######################################################


CONCEPTS = [
    {
        'class_data_dir': CLASS_DATA_DIR,
        'class_guidance_scale': CLASSIFICATION_CFG_SCALE,
        'class_infer_steps': CLASSIFICATION_STEPS,
        'class_negative_prompt': "",
        'class_prompt': CLASS_PROMPT,
        'class_token': "",
        'instance_data_dir': INSTANCE_DATA_DIR,
        'instance_prompt': INSTANCE_PROMPT,
        'instance_token': "",
        'is_valid': True,
        'n_save_sample': NUMBER_OF_SAMPLES_TO_GENERATE,
        'num_class_images_per': CLASS_IMAGES_PER_INSTANCE_IMAGE,
        'sample_seed': SAMPLE_SEED,
        'save_guidance_scale': SAMPLE_CFG_SCALE,
        'save_infer_steps': SAMPLE_STEPS,
        'save_sample_negative_prompt': "",
        'save_sample_prompt': SAMPLE_IMAGE_PROMPT,
        'save_sample_template': ""
    }
]

PAYLOAD = {
    'weight_decay': 0.01,
    'attention': ATTENTION,
    'cache_latents': CACHE_LATENTS,
    'clip_skip': 1,
    'concepts_list': CONCEPTS,
    'concepts_path': '',
    'custom_model_name': '',
    'deterministic': False,
    'disable_class_matching': False,
    'disable_logging': False,
    'ema_predict': False,
    'epoch': 0,
    'epoch_pause_frequency': 0,
    'epoch_pause_time': 0,
    'freeze_clip_normalization': False,
    'gradient_accumulation_steps': 1,
    'gradient_checkpointing': GRADIENT_CHECKPOINTING,
    'gradient_set_to_none': True,
    'graph_smoothing': 50,
    'half_model': False,
    'has_ema': HAS_EMA,
    'hflip': False,
    'infer_ema': False,
    'initial_revision': 0,
    'learning_rate': LEARNING_RATE,
    'learning_rate_min': 0.000001,
    'lifetime_revision': 0,
    'lora_learning_rate': 0.0001,
    'lora_model_name': '',
    'lora_txt_learning_rate': 0.00005,
    'lora_txt_rank': 4,
    'lora_txt_weight': 1,
    'lora_unet_rank': 4,
    'lora_weight': 1,
    'lora_use_buggy_requires_grad': False,
    'lr_cycles': 1,
    'lr_factor': LEARNING_RATE_CONSTANT_LINEAR_STARTING_FACTOR,
    'lr_power': 1,
    'lr_scale_pos': LEARNING_RATE_SCALE_POSITION,
    'lr_scheduler': LEARNING_RATE_SCHEDULER,
    'lr_warmup_steps': 0,
    'max_token_length': 75,
    'mixed_precision': MIXED_PRECISION,
    'model_dir': f'{WEBUI_INSTALLATION_PATH}/models/dreambooth/{MODEL_NAME}',
    'model_name': MODEL_NAME,
    'model_path': f'{WEBUI_INSTALLATION_PATH}/models/dreambooth/{MODEL_NAME}',
    'noise_scheduler': 'DDPM',
    'num_train_epochs': NUM_TRAIN_EPOCHS,
    'offset_noise': 0,
    'optimizer': OPTIMIZER,
    'pad_tokens': True,
    'pretrained_model_name_or_path': f'{WEBUI_INSTALLATION_PATH}/models/dreambooth/{MODEL_NAME}',
    'pretrained_vae_name_or_path': '',
    'prior_loss_scale': False,
    'prior_loss_target': 100,
    'prior_loss_weight': 0.75,
    'prior_loss_weight_min': 0.1,
    'resolution': RESOLUTION,
    'revision': 0,
    'sample_batch_size': 1,
    'sanity_prompt': SANITY_PROMPT,
    'sanity_seed': 420420,
    'save_ckpt_after': SAVE_CKPT_AFTER,
    'save_ckpt_cancel': SAVE_CKPT_CANCEL,
    'save_ckpt_during': SAVE_CKPT_DURING,
    'save_ema': SAVE_EMA,
    'save_embedding_every': SAVE_EMBEDDING_EVERY,
    'save_lora_after': True,
    'save_lora_cancel': False,
    'save_lora_during': False,
    'save_lora_for_extra_net': False,
    'save_preview_every': SAVE_PREVIEW_EVERY,
    'save_safetensors': SAVE_SAFETENSORS,
    'save_state_after': False,
    'save_state_cancel': False,
    'save_state_during': False,
    'scheduler': SCHEDULER,
    'shared_diffusers_path': '',
    'shuffle_tags': False,
    'snapshot': '',
    'split_loss': True,
    'src': SOURCE_MODEL_PATH,
    'stop_text_encoder': STOP_TEXT_ENCODER,
    'strict_tokens': False,
    'dynamic_img_norm': False,
    'tenc_weight_decay': 0.01,
    'tenc_grad_clip_norm': 0,
    'tomesd': 0,
    'train_batch_size': 1,
    'train_imagic': False,
    'train_unet': True,
    'train_unfrozen': TRAIN_UNFROZEN,
    'txt_learning_rate': TXT_LEARNING_RATE,
    'use_concepts': False,
    'use_ema': USE_EMA,
    'use_lora': False,
    'use_lora_extended': False,
    'use_shared_src': False,
    'use_subdir': SAVE_CHECKPOINT_TO_SUBDIRECTORY,
    'v2': False
}


def set_model_config():
    endpoint = f'{URL}/dreambooth/model_config'

    r = requests.post(
        endpoint,
        json=PAYLOAD
    )

    print(r.status_code)
    response_text = r.json()
    print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    set_model_config()
