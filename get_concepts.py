#!/usr/bin/env python3
import json
import requests

# FIXME: TypeError: Object of type Concept is not JSON serializable

URL = 'http://172.30.61.140:7860'
MODEL_NAME = 'PROTOGEN-TEST'

def get_concepts():
    endpoint = f'{URL}/dreambooth/concepts'
    endpoint += f'?model_name={MODEL_NAME}'

    r = requests.get(
        endpoint
    )

    print(r.status_code)
    response_text = r.json()
    print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    get_concepts()
