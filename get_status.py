#!/usr/bin/env python3
import json
import requests
import util


def get_status():
    url = config['webui_url'].rstrip('/')
    endpoint = f'{url}/dreambooth/status'

    r = requests.get(
        endpoint
    )

    print(r.status_code)
    response_text = r.json()
    current_state = json.loads(response_text['current_state'])
    print(json.dumps(current_state, indent=4, default=str))


if __name__ == '__main__':
    config = util.load_config()
    get_status()
