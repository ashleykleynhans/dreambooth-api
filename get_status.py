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

    resp_json = r.json()

    if r.status_code == 200:
        current_state = resp_json['current_state']
        print(json.dumps(current_state, indent=4, default=str))
    else:
        print(r.status_code)
        print(json.dumps(resp_json, indent=4, default=str))


if __name__ == '__main__':
    config = util.load_config()
    get_status()
