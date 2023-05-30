#!/usr/bin/env python3
import json
import requests
import util

# TODO: Not currently working due to a bug in resolving
# the model path - see https://github.com/d8ahazard/sd_dreambooth_extension/issues/1230


def delete_model():
    url = config['webui_url'].rstrip('/')
    model_name = config['new_model_name']

    endpoint = f'{url}/dreambooth/model'

    payload = {
        'model_name': model_name
    }

    r = requests.delete(
        endpoint,
        data=payload
    )

    if r.status_code == 200:
        print(r.status_code)
        print(r.content)
    else:
        print(r.status_code)
        json_resp = r.json()
        print(json.dumps(json_resp, indent=4, default=str))


if __name__ == '__main__':
    config = util.load_config()
    delete_model()
