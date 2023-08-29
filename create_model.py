#!/usr/bin/env python3
import json
import requests
import util

MODEL_TYPE = 'v1x'
TRAIN_UNFROZEN = True
EXTRACT_EMA = False


def create_model():
    url = config['webui_url'].rstrip('/')
    new_model_name = config['new_model_name']
    model_path = config['model_path']
    source_model = config['source_model']
    new_model_scheduler = config['new_model_scheduler']

    endpoint = f'{url}/dreambooth/createModel'
    endpoint += f'?new_model_name={new_model_name}'
    endpoint += f'&new_model_src={model_path}/{source_model}'
    endpoint += f'&new_model_scheduler={new_model_scheduler}'
    endpoint += f'&model_type={MODEL_TYPE.lower()}'
    endpoint += f'&train_unfrozen={str(TRAIN_UNFROZEN).lower()}'
    endpoint += f'&new_model_extract_ema={str(EXTRACT_EMA).lower()}'

    r = requests.post(
        endpoint
    )

    response_text = r.json()

    if r.status_code == 200:
        print(response_text)
    else:
        print(r.status_code)
        print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    config = util.load_config()
    create_model()
