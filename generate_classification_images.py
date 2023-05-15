#!/usr/bin/env python3
import json
import requests

URL = 'http://172.30.61.140:7860'
MODEL_NAME = 'PROTOGEN-TEST'
IMAGE_GENERATION_LIBRARY = ''


def generate_classification_images():
    endpoint = f'{URL}/dreambooth/classifiers'

    payload = {
        'model_name':  MODEL_NAME,
        # 'class_gen_method': IMAGE_GENERATION_LIBRARY
    }

    r = requests.post(
        endpoint,
        data=payload
    )

    print(r.status_code)

    if r.status_code == 200:
        resp_json = r.json()
        print(resp_json['message'])
    else:
        resp_json = r.json()
        print(json.dumps(resp_json, indent=4, default=str))


if __name__ == '__main__':
    generate_classification_images()
