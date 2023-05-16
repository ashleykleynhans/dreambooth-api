#!/usr/bin/env python3
import json
import requests

# FIXME: This script is not working

URL = 'http://172.30.61.140:7860'
MODEL_NAME = 'PROTOGEN-TEST'
CONCEPT_INDEX = -1


def get_zip_file(response):
    content_disposition = response.headers.get('content-disposition')

    if content_disposition is None:
        resp_json = response.json()
        print(resp_json['message'])
    else:
        filename = content_disposition.split('=')[1]

        with open(filename, 'wb') as f:
            print(f'Saving: {filename}')
            f.write(response.content)


def get_classification_images():
    endpoint = f'{URL}/dreambooth/classifiers'
    endpoint += f'?model_name={MODEL_NAME}'
    endpoint += f'concept_idx={CONCEPT_INDEX}'

    r = requests.get(
        endpoint
    )

    print(r.status_code)

    if r.status_code == 200:
        print(r.headers)
        # content_type = r.headers.get('content-type')
        #
        # if content_type == '':
        #     get_zip_file(r)
        # else:
        #     resp_json = r.json()
        #     print(resp_json['message'])
    else:
        resp_json = r.json()
        print(json.dumps(resp_json, indent=4, default=str))


if __name__ == '__main__':
    get_classification_images()
