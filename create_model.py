#!/usr/bin/env python3
import requests
import re

URL = 'http://172.30.61.140:7860'
NEW_MODEL_NAME = 'test-model'
SRC_MODEL = 'v1-5-pruned.safetensors'
TRAIN_UNFROZEN = 'false'


def create_model():
    endpoint = f'{URL}/dreambooth/createModel'
    endpoint += f'?new_model_name={NEW_MODEL_NAME}'
    endpoint += f'&new_model_src={SRC_MODEL}'
    endpoint += f'&train_unfrozen={TRAIN_UNFROZEN}'

    r = requests.post(
        endpoint
    )

    response_text = r.json()
    match = re.findall('Checkpoint successfully extracted to (.*)', response_text)

    if match:
        print(f'Model path: {match[0]}')


if __name__ == '__main__':
    create_model()
