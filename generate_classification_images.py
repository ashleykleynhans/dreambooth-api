#!/usr/bin/env python3
import json
import requests
import util


def generate_classification_images():
    url = config['webui_url'].rstrip('/')
    model_name = config['new_model_name']
    endpoint = f'{url}/dreambooth/classifiers'

    payload = {
        'model_name':  model_name,
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
    config = util.load_config()
    generate_classification_images()
