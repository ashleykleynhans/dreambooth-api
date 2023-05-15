#!/usr/bin/env python3
import json
import requests

URL = 'http://172.30.61.140:7860'

MODEL_NAME = 'PROTOGEN-TEST'


def start_training():
    endpoint = f'{URL}/dreambooth/start_training'
    endpoint += f'?model_name={MODEL_NAME}'
    endpoint += f'&use_tx2img=false'

    r = requests.post(
        endpoint
    )

    print(r.status_code)
    response_text = r.json()
    print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    start_training()
