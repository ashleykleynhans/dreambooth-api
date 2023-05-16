#!/usr/bin/env python3
import requests
import re

URL = 'http://172.30.61.140:7860'
NEW_MODEL_NAME = 'test-model'
SRC_MODEL = 'v1-5-pruned.safetensors'
SHARED_DIFFUSERS_SOURCE = ''
CREATE_FROM_HUB = 'false'
NEW_MODEL_HUB_URL = ''
IS_512_RESOLUTION = 'true'
TRAIN_UNFROZEN = 'false'
HUGGINGFACE_HUB_TOKEN = ''
NEW_MODEL_EXTRACT_EMA = 'false'


def create_model():
    endpoint = f'{URL}/dreambooth/createModel'
    endpoint += f'?new_model_name={NEW_MODEL_NAME}'
    endpoint += f'&new_model_src={SRC_MODEL}'
    # endpoint += f'&new_model_shared_src={SHARED_DIFFUSERS_SOURCE}'
    endpoint += f'&create_from_hub={CREATE_FROM_HUB}'
    # endpoint += f'&new_model_url={NEW_MODEL_HUB_URL}'
    endpoint += f'&is_512={IS_512_RESOLUTION}'
    endpoint += f'&train_unfrozen={TRAIN_UNFROZEN}'
    # endpoint += f'&new_model_token={HUGGINGFACE_HUB_TOKEN}'
    endpoint += f'&new_model_extract_ema={NEW_MODEL_EXTRACT_EMA}'

    r = requests.post(
        endpoint
    )

    response_text = r.json()
    match = re.findall('Checkpoint successfully extracted to (.*)', response_text)

    if match:
        print(f'Model path: {match[0]}')


if __name__ == '__main__':
    create_model()
