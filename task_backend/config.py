import os
import json


def load_config():
    if os.path.exists('./remote_config.json'):
        with open('./remote_config.json', 'r') as i:
            return json.load(i)

    if 'CONFIG_REMOTE_PATH' not in os.environ:
        path = '/etc/config/remote_config.json'
    else:
        path = os.environ['CONFIG_REMOTE_PATH']

    if not os.path.exists(path):
        print("Cannot load config. Exit.")
        exit()

    with open(path, 'r') as i:
        return json.load(i)
