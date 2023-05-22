#!/usr/bin/env python3
import requests
import util


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


def get_checkpoint():
    url = config['webui_url']
    model_name = config['new_model_name']

    endpoint = f'{url}/dreambooth/checkpoint'
    endpoint += f'?model_name={model_name}'

    r = requests.get(
        endpoint
    )

    if r.status_code == 200:
        content_type = r.headers.get('content-type')

        if content_type == '':
            get_zip_file(r)
        else:
            resp_json = r.json()
            print(resp_json)
    else:
        resp_json = r.json()
        print(resp_json['message'])


if __name__ == '__main__':
    config = util.load_config()
    get_checkpoint()
