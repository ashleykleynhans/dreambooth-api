#!/usr/bin/env python3
import json
import requests
from enum import Enum


class LearningRateScheduler(Enum):
    CONSTANT = 'constant'
    CONSTANT_WITH_WARMUP = 'constant_with_warmup'
    COSINE = 'cosine'
    COSINE_ANNEALING = 'cosine_annealing'
    COSINE_ANNEALING_WITH_RESTARTS = 'cosine_annealing_with_restarts'
    COSINE_WITH_RESTARTS = 'cosine_with_restarts'
    DEDAPT_WITH_WARMUP = 'dadapt_with_warmup'
    LINEAR = 'linear'
    LINEAR_WITH_WARMUP = 'linear_with_warmup'
    POLYNOMIAL = 'polynomial'


class Scheduler(Enum):
    DDIM = 'DDIM'
    DDPM = 'DDPM'
    PNDM = 'PNDM'
    LMSDiscrete = 'LMSDiscrete'
    EulerDiscrete = 'EulerDiscrete'
    HeunDiscrete = 'HeunDiscrete'
    EulerAncestralDiscrete = 'EulerAncestralDiscrete'
    DPMSolverMultistep = 'DPMSolverMultistep'
    DPMSolverSinglestep = 'DPMSolverSinglestep'
    KDPM2Discrete = 'KDPM2Discrete'
    KDPM2AncestralDiscrete = 'KDPM2AncestralDiscrete'
    DEISMultistep = 'DEISMultistep'
    UniPCMultistep = 'UniPCMultistep'


class NoiseScheduler(Enum):
    DDPM = 'DDPM'
    DEIS = 'DEIS'
    UniPC = 'UniPC'


class Optimizer(Enum):
    AdamW_Torch = 'Torch AdamW'
    AdamW_8bit = '8bit AdamW'
    Lion = 'Lion'
    AdamW_Dadaptation = 'AdamW Dadaptation'
    Adan_Dadaptation = 'Adan Dadaptation'
    AdanIP_Dadaptation = 'AdanIP Dadaptation'


URL = 'http://172.17.1.140:7860'
WEBUI_INSTALLATION_PATH = '/home/ubuntu/stable-diffusion-webui'

#######################################################
# Model
#######################################################
MODEL_NAME = 'test-model'
SOURCE_CHECKPOINT = '/models/ckpt/v1-5-pruned.safetensors'
MODEL_REVISION = 0
MODEL_EPOCH = 0
V2_MODEL = False
HAS_EMA = False
TRAIN_UNFROZEN = True
SCHEDULER = Scheduler.DDPM.value

#######################################################
# General
#######################################################
USE_LORA = False
USE_LORA_EXTENDED = False
TRAIN_IMAGIC_ONLY = False

#######################################################
# Intervals
#######################################################
TRAINING_STEPS_PER_IMAGE = 200
PAUSE_AFTER_N_EPOCHS = 0
AMOUNT_OF_TIME_TO_PAUSE_BETWEEN_EPOCHS = 0
SAVE_MODEL_FREQUENCY = 10
SAVE_PREVIEW_FREQUENCY = 0

#######################################################
# Batching
#######################################################
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
CLASS_BATCH_SIZE = 1
SET_GRADIENT_TO_NONE_WHEN_ZEROING = True
GRADIENT_CHECKPOINTING = False

#######################################################
# Learning Rate
#######################################################
LEARNING_RATE = 0.0000001
TEXT_ENCODER_LEARNING_RATE = 0.0000001
LORA_UNET_LEARNING_RATE = 0.000001
LORA_TEXT_ENCODER_LEARNING_RATE = 0.000005
LEARNING_RATE_SCHEDULER = LearningRateScheduler.CONSTANT_WITH_WARMUP.value
#######################################################
# Constant with Warmup Learning Rate Settings
#######################################################
LEARNING_RATE_WARMUP_STEPS = 0
#######################################################
# OR Constant Learning Rate Settings
#######################################################
LEARNING_RATE_CONSTANT_LINEAR_STARTING_FACTOR = 1
LEARNING_RATE_SCALE_POSITION = 1
#######################################################

#######################################################
# Image Processing
#######################################################
MAX_RESOLUTION = 512
APPLY_HORIZONTAL_FLIP = False
DYNAMIC_IMAGE_NORMALIZATION = False
USE_EMA = True
OPTIMIZER = Optimizer.Lion.value
MIXED_PRECISION = 'fp16'
MEMORY_ATTENTION = 'default'
CACHE_LATENTS = True
TRAIN_UNET = True
STEP_RATIO_OF_TEXT_ENCODER_TRAINING = 0.75
OFFSET_NOISE = 0
FREEZE_CLIP_NORMALIZATION_LAYERS = True
CLIP_SKIP = 1
WEIGHT_DECAY = 0.1
TENC_WEIGHT_DECAY = 0.1
TENC_GRADIENT_CLIP_NORM = 0
PAD_TOKENS = True
STRICT_TOKENS = False
SHUFFLE_TAGS = False
MAX_TOKEN_LENGTH = 75

#######################################################
# Prior Loss
#######################################################
SCALE_PRIOR_LOSS = False
PRIOR_LOSS_WEIGHT = 0.75
PRIOR_LOSS_TARGET = 100
MINIMUM_PRIOR_LOSS_WEIGHT = 0.1

#######################################################
# Sanity Samples
#######################################################
SANITY_SAMPLE_PROMPT = 'photo of ohwx woman by Tomer Hanuka'
# Negative sanity sample prompt does not appear to be supported by API
# SANITY_SAMPLE_NEGATIVE_PROMPT = ''
SANITY_SAMPLE_SEED = 420420

#######################################################
# Miscellaneous
#######################################################
PRETRAINED_VAE_NAME_OR_PATH = ''
USE_CONCEPTS_LIST = False
CONCEPTS_LIST_PATH = ''

#######################################################
# Concept 1 Configuration (can add up to 3)
#######################################################
# Directories
#######################################################
DATASET_DIRECTORY = '/home/ubuntu/training-512x512'
CLASSIFICATION_DATASET_DIRECTORY = '/home/ubuntu/woman-images'
#######################################################
# Filewords
#######################################################
INSTANCE_TOKEN = ''
CLASS_TOKEN = ''
#######################################################
# Training Prompts
#######################################################
INSTANCE_PROMPT = 'ohwx woman'
CLASS_PROMPT = 'photo of woman'
CLASSIFICATION_IMAGE_NEGATIVE_PROMPT = ''
#######################################################
# Sample Prompts
#######################################################
SAMPLE_IMAGE_PROMPT = 'photo of ohwx woman'
SAMPLE_NEGATIVE_PROMPT = ''
SAMPLE_PROMPT_TEMPLATE_FILE = ''
#######################################################
# Class Image Generation
#######################################################
CLASS_IMAGES_PER_INSTANCE_IMAGE = 50
CLASSIFICATION_CFG_SCALE = 7.5
CLASSIFICATION_STEPS = 40
#######################################################
# Sample Image Generation
#######################################################
NUMBER_OF_SAMPLES_TO_GENERATE = 1
SAMPLE_SEED = -1
SAMPLE_CFG_SCALE = 7.5
SAMPLE_STEPS = 20
#######################################################

#######################################################
# Saving
#######################################################
# General
#######################################################
CUSTOM_MODEL_NAME = ''
SAVE_EMA_WEIGHTS_TO_GENERATED_MODELS = False
USE_EMA_WEIGHTS_FOR_INFERENCE = False
#######################################################
# Checkpoints
#######################################################
HALF_MODEL = False
SAVE_CHECKPOINT_TO_SUBDIRECTORY = True
GENERATE_CKPT_DURING_TRAINING = True
GENERATE_CKPT_WHEN_TRAINING_COMPLETES = True
GENERATE_CKPT_WHEN_TRAINING_IS_CANCELED = False
SAVE_SAFETENSORS = True
#######################################################
# Lora
#######################################################
LORA_UNET_RANK = 4
LORA_TEXT_ENCODER_RANK = 4
LORA_WEIGHT = 1
LORA_TEXT_WEIGHT = 1
GENERATE_LORA_WEIGHTS_WHEN_SAVING_DURING_TRAINING = False
GENERATE_LORA_WEIGHTS_WHEN_TRAINING_COMPLETES = True
GENERATE_LORA_WEIGHTS_WHEN_TRAINING_IS_CANCELED = False
GENERATE_LORA_WEIGHTS_FOR_EXTRA_NETWORKS = False
#######################################################
# Diffusion Weights (training snapshots)
#######################################################
SAVE_SEPARATE_DIFFUSERS_SNAPSHOTS_WHEN_SAVING_DURING_TRAINING = False
SAVE_SEPARATE_DIFFUSERS_SNAPSHOTS_WHEN_TRAINING_COMPLETES = False
SAVE_SEPARATE_DIFFUSERS_SNAPSHOTS_WHEN_TRAINING_IS_CANCELED = False

#######################################################
# Experimental Settings
#######################################################
TOKEN_MERGING = 0
CALCULATE_SPLIT_LOSS = True
DISABLE_CLASS_MATCHING = False
DISABLE_LOGGING = True
DETERMINISTIC = True
USE_EMA_FOR_PREDICTION = False
LORA_USE_BUGGY_REQUIRES_GRAD = False
NOISE_SCHEDULER = NoiseScheduler.DEIS.value


CONCEPTS = [
    {
        'class_data_dir': CLASSIFICATION_DATASET_DIRECTORY,
        'class_guidance_scale': CLASSIFICATION_CFG_SCALE,
        'class_infer_steps': CLASSIFICATION_STEPS,
        'class_negative_prompt': CLASSIFICATION_IMAGE_NEGATIVE_PROMPT,
        'class_prompt': CLASS_PROMPT,
        'class_token': CLASS_TOKEN,
        'instance_data_dir': DATASET_DIRECTORY,
        'instance_prompt': INSTANCE_PROMPT,
        'instance_token': INSTANCE_TOKEN,
        'is_valid': True,
        'n_save_sample': NUMBER_OF_SAMPLES_TO_GENERATE,
        'num_class_images_per': CLASS_IMAGES_PER_INSTANCE_IMAGE,
        'sample_seed': SAMPLE_SEED,
        'save_guidance_scale': SAMPLE_CFG_SCALE,
        'save_infer_steps': SAMPLE_STEPS,
        'save_sample_negative_prompt': SAMPLE_NEGATIVE_PROMPT,
        'save_sample_prompt': SAMPLE_IMAGE_PROMPT,
        'save_sample_template': SAMPLE_PROMPT_TEMPLATE_FILE
    }
]

PAYLOAD = {
    'weight_decay': WEIGHT_DECAY,
    'attention': MEMORY_ATTENTION,
    'cache_latents': CACHE_LATENTS,
    'clip_skip': CLIP_SKIP,
    'concepts_list': CONCEPTS,
    'concepts_path': CONCEPTS_LIST_PATH,
    'custom_model_name': CUSTOM_MODEL_NAME,
    'deterministic': DETERMINISTIC,
    'disable_class_matching': DISABLE_CLASS_MATCHING,
    'disable_logging': DISABLE_LOGGING,
    'ema_predict': USE_EMA_FOR_PREDICTION,
    'epoch': MODEL_EPOCH,
    'epoch_pause_frequency': PAUSE_AFTER_N_EPOCHS,
    'epoch_pause_time': AMOUNT_OF_TIME_TO_PAUSE_BETWEEN_EPOCHS,
    'freeze_clip_normalization': FREEZE_CLIP_NORMALIZATION_LAYERS,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    'gradient_checkpointing': GRADIENT_CHECKPOINTING,
    'gradient_set_to_none': SET_GRADIENT_TO_NONE_WHEN_ZEROING,
    'graph_smoothing': 50,
    'half_model': HALF_MODEL,
    'has_ema': HAS_EMA,
    'hflip': APPLY_HORIZONTAL_FLIP,
    'infer_ema': USE_EMA_WEIGHTS_FOR_INFERENCE,
    'initial_revision': MODEL_REVISION,
    'learning_rate': LEARNING_RATE,
    'learning_rate_min': 0.000001,
    'lifetime_revision': MODEL_REVISION,
    'lora_learning_rate': LORA_UNET_LEARNING_RATE,
    'lora_model_name': '',
    'lora_txt_learning_rate': LORA_TEXT_ENCODER_LEARNING_RATE,
    'lora_txt_rank': LORA_TEXT_ENCODER_RANK,
    'lora_txt_weight': LORA_TEXT_WEIGHT,
    'lora_unet_rank': LORA_UNET_RANK,
    'lora_weight': LORA_WEIGHT,
    'lora_use_buggy_requires_grad': LORA_USE_BUGGY_REQUIRES_GRAD,
    'lr_cycles': 1,
    'lr_factor': LEARNING_RATE_CONSTANT_LINEAR_STARTING_FACTOR,
    'lr_power': 1,
    'lr_scale_pos': LEARNING_RATE_SCALE_POSITION,
    'lr_scheduler': LEARNING_RATE_SCHEDULER,
    'lr_warmup_steps': LEARNING_RATE_WARMUP_STEPS,
    'max_token_length': MAX_TOKEN_LENGTH,
    'mixed_precision': MIXED_PRECISION,
    'model_dir': f'{WEBUI_INSTALLATION_PATH}/models/dreambooth/{MODEL_NAME}',
    'model_name': MODEL_NAME,
    'model_path': f'{WEBUI_INSTALLATION_PATH}/models/dreambooth/{MODEL_NAME}',
    'noise_scheduler': NOISE_SCHEDULER,
    'num_train_epochs': TRAINING_STEPS_PER_IMAGE,
    'offset_noise': OFFSET_NOISE,
    'optimizer': OPTIMIZER,
    'pad_tokens': PAD_TOKENS,
    'pretrained_model_name_or_path': f'{WEBUI_INSTALLATION_PATH}/models/dreambooth/{MODEL_NAME}',
    'pretrained_vae_name_or_path': PRETRAINED_VAE_NAME_OR_PATH,
    'prior_loss_scale': SCALE_PRIOR_LOSS,
    'prior_loss_target': PRIOR_LOSS_TARGET,
    'prior_loss_weight': PRIOR_LOSS_WEIGHT,
    'prior_loss_weight_min': MINIMUM_PRIOR_LOSS_WEIGHT,
    'resolution': MAX_RESOLUTION,
    'revision': MODEL_REVISION,
    'sample_batch_size': CLASS_BATCH_SIZE,
    'sanity_prompt': SANITY_SAMPLE_PROMPT,
    'sanity_seed': SANITY_SAMPLE_SEED,
    'save_ckpt_after': GENERATE_CKPT_WHEN_TRAINING_COMPLETES,
    'save_ckpt_cancel': GENERATE_CKPT_WHEN_TRAINING_IS_CANCELED,
    'save_ckpt_during': GENERATE_CKPT_DURING_TRAINING,
    'save_ema': SAVE_EMA_WEIGHTS_TO_GENERATED_MODELS,
    'save_embedding_every': SAVE_MODEL_FREQUENCY,
    'save_lora_after': GENERATE_LORA_WEIGHTS_WHEN_TRAINING_COMPLETES,
    'save_lora_cancel': GENERATE_LORA_WEIGHTS_WHEN_TRAINING_IS_CANCELED,
    'save_lora_during': GENERATE_LORA_WEIGHTS_WHEN_SAVING_DURING_TRAINING,
    'save_lora_for_extra_net': GENERATE_LORA_WEIGHTS_FOR_EXTRA_NETWORKS,
    'save_preview_every': SAVE_PREVIEW_FREQUENCY,
    'save_safetensors': SAVE_SAFETENSORS,
    'save_state_after': SAVE_SEPARATE_DIFFUSERS_SNAPSHOTS_WHEN_TRAINING_COMPLETES,
    'save_state_cancel': SAVE_SEPARATE_DIFFUSERS_SNAPSHOTS_WHEN_TRAINING_IS_CANCELED,
    'save_state_during': SAVE_SEPARATE_DIFFUSERS_SNAPSHOTS_WHEN_SAVING_DURING_TRAINING,
    'scheduler': SCHEDULER,
    'shared_diffusers_path': '',
    'shuffle_tags': SHUFFLE_TAGS,
    'snapshot': '',
    'split_loss': CALCULATE_SPLIT_LOSS,
    'src': SOURCE_CHECKPOINT,
    'stop_text_encoder': STEP_RATIO_OF_TEXT_ENCODER_TRAINING,
    'strict_tokens': STRICT_TOKENS,
    'dynamic_img_norm': DYNAMIC_IMAGE_NORMALIZATION,
    'tenc_weight_decay': TENC_WEIGHT_DECAY,
    'tenc_grad_clip_norm': TENC_GRADIENT_CLIP_NORM,
    'tomesd': TOKEN_MERGING,
    'train_batch_size': BATCH_SIZE,
    'train_imagic': TRAIN_IMAGIC_ONLY,
    'train_unet': TRAIN_UNET,
    'train_unfrozen': TRAIN_UNFROZEN,
    'txt_learning_rate': TEXT_ENCODER_LEARNING_RATE,
    'use_concepts': USE_CONCEPTS_LIST,
    'use_ema': USE_EMA,
    'use_lora': USE_LORA,
    'use_lora_extended': USE_LORA_EXTENDED,
    'use_shared_src': False,
    'use_subdir': SAVE_CHECKPOINT_TO_SUBDIRECTORY,
    'v2': V2_MODEL
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
