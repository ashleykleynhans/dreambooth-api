#!/usr/bin/env python3
import json
import requests

URL = 'http://172.17.1.140:7860'


def get_status():
    endpoint = f'{URL}/dreambooth/status'

    r = requests.get(
        endpoint
    )

    print(r.status_code)
    response_text = r.json()
    current_state = json.loads(response_text['current_state'])
    print(json.dumps(current_state, indent=4, default=str))


if __name__ == '__main__':
    get_status()
