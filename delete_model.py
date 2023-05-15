#!/usr/bin/env python3
import requests

# FIXME: Not sure what this is trying to delete, because
# it returns an HTTP 404 Not Found
URL = 'http://172.30.61.140:7860'
MODEL_NAME = 'PROTOGEN-TEST'


def delete_model():
    endpoint = f'{URL}/dreambooth/mopel'
    endpoint += f'?model_name={MODEL_NAME}'

    r = requests.delete(
        endpoint
    )

    print(r.status_code)
    print(r.content)


if __name__ == '__main__':
    delete_model()
