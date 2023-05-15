#!/usr/bin/env python3
import requests
URL = 'http://172.30.61.140:7860'


def get_status():
    endpoint = f'{URL}/dreambooth/status_images'

    r = requests.get(
        endpoint
    )

    if r.status_code == 200:
        filename = r.headers.get('content-disposition').split('=')[1]

        with open(filename, 'wb') as f:
            print(f'Saving: {filename}')
            f.write(r.content)


if __name__ == '__main__':
    get_status()
