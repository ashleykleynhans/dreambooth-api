#!/usr/bin/env python3
import requests

URL = 'http://172.17.1.140:7860'
MODEL_NAME = 'test-model'


def start_training():
    endpoint = f'{URL}/dreambooth/start_training'
    endpoint += f'?model_name={MODEL_NAME}'
    endpoint += f'&use_tx2img=false'

    r = requests.post(
        endpoint
    )

    print(r.status_code)

    if r.status_code == 200:
        response_text = r.json()
        print(response_text['Status'])


if __name__ == '__main__':
    start_training()
