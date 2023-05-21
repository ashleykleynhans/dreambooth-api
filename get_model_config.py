#!/usr/bin/env python3
import json
import requests

URL = 'http://172.17.1.140:7860'
MODEL_NAME = 'test-model'


def get_model_config():
    endpoint = f'{URL}/dreambooth/model_config'
    endpoint += f'?model_name={MODEL_NAME}'

    r = requests.get(
        endpoint
    )

    print(r.status_code)
    response_text = r.json()
    print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    get_model_config()
