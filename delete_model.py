#!/usr/bin/env python3
import json
import requests

# FIXME: Not sure what this is trying to delete, because
# it returns an HTTP 404 Not Found
URL = 'http://172.30.61.140:7860'
MODEL_NAME = 'test-model'


def delete_model():
    endpoint = f'{URL}/dreambooth/mopel'
    endpoint += f'?model_name={MODEL_NAME}'

    r = requests.delete(
        endpoint
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
