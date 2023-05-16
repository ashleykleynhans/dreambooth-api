#!/usr/bin/env python3
import json
import requests

# TODO: Not currently working due to a bug in resolving
# the model path - see https://github.com/d8ahazard/sd_dreambooth_extension/issues/1230

URL = 'http://172.30.61.140:7860'
MODEL_NAME = 'test-model'


def delete_model():
    endpoint = f'{URL}/dreambooth/model'

    payload = {
        'model_name': MODEL_NAME
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
    delete_model()
