#!/usr/bin/env python3
import json
import requests

URL = 'http://172.17.1.140:7860'


def get_models():
    endpoint = f'{URL}/dreambooth/models'

    r = requests.get(
        endpoint
    )

    print(r.status_code)
    response_text = r.json()
    print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    get_models()
