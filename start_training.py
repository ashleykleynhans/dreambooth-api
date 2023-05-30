#!/usr/bin/env python3
import requests
import util


def start_training():
    url = config['webui_url'].rstrip('/')
    model_name = config['new_model_name']

    endpoint = f'{url}/dreambooth/start_training'
    endpoint += f'?model_name={model_name}'
    endpoint += f'&use_tx2img=false'

    r = requests.post(
        endpoint
    )

    print(r.status_code)

    if r.status_code == 200:
        response_text = r.json()
        print(response_text['Status'])


if __name__ == '__main__':
    config = util.load_config()
    start_training()
