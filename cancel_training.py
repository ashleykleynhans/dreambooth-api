#!/usr/bin/env python3
import requests

URL = 'http://172.17.1.140:7860'


def cancel_training():
    endpoint = f'{URL}/dreambooth/cancel'

    r = requests.get(
        endpoint
    )

    print(r.status_code)

    if r.status_code == 200:
        response_text = r.json()
        print(response_text['message'])


if __name__ == '__main__':
    cancel_training()
