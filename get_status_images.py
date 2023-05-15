#!/usr/bin/env python3
import requests

URL = 'http://172.30.61.140:7860'


def get_image(response):
    filename = 'sample.png'

    with open(filename, 'wb') as f:
        print(f'Saving: {filename}')
        f.write(response.content)


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


def get_status():
    endpoint = f'{URL}/dreambooth/status_images'

    r = requests.get(
        endpoint
    )

    if r.status_code == 200:
        content_type = r.headers.get('content-type')

        if content_type == 'image/png':
            get_image(r)
        elif content_type == '':
            get_zip_file(r)
        else:
            resp_json = r.json()
            print(resp_json['message'])


if __name__ == '__main__':
    get_status()
