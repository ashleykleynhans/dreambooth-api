#!/usr/bin/env python3
import json
import requests
import util


def get_lora_models():
    url = config['webui_url'].rstrip('/')
    model_name = config['new_model_name']

    endpoint = f'{url}/dreambooth/models_lora'
    endpoint += f'?model_name={model_name}'

    r = requests.get(
        endpoint
    )

    print(r.status_code)
    response_text = r.json()
    print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    config = util.load_config()
    get_lora_models()
