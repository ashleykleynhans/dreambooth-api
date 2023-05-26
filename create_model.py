#!/usr/bin/env python3
import json
import requests
import re
import util


def create_model():
    url = config['webui_url']
    new_model_name = config['new_model_name']
    source_model = config['source_model']
    new_model_scheduler = config['new_model_scheduler']

    endpoint = f'{url}/dreambooth/createModel'
    endpoint += f'?new_model_name={new_model_name}'
    endpoint += f'&new_model_src={source_model}'
    endpoint += f'&new_model_scheduler={new_model_scheduler}'
    endpoint += f'&is_512=true'
    endpoint += '&train_unfrozen=true'
    endpoint += '&new_model_extract_ema=false'

    r = requests.post(
        endpoint
    )

    response_text = r.json()

    if r.status_code == 200:
        match = re.findall('Checkpoint successfully extracted to (.*)', response_text)

        if match:
            print(f'Model path: {match[0]}')
    else:
        print(r.status_code)
        print(json.dumps(response_text, indent=4, default=str))


if __name__ == '__main__':
    config = util.load_config()
    create_model()
