#!/usr/bin/env python3
import requests
import util


def cancel_training():
    url = config['webui_url']
    endpoint = f'{url}/dreambooth/cancel'

    r = requests.get(
        endpoint
    )

    print(r.status_code)

    if r.status_code == 200:
        response_text = r.json()
        print(response_text['message'])


if __name__ == '__main__':
    config = util.load_config()
    cancel_training()
