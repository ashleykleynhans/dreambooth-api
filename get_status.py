#!/usr/bin/env python3
import json
import requests
import re
URL = 'http://172.30.61.140:7860'


def get_status():
    endpoint = f'{URL}/dreambooth/status'

    r = requests.get(
        endpoint
    )

    response_text = r.json()
    print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    get_status()
