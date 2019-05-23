import os
import json


def load_config(failing_default=None):
    base = os.path.dirname(__file__)
    path = os.path.join(base, './remote_config.json')

    if os.path.exists(path):
        with open(path, 'r') as i:
            return json.load(i)

    if 'CONFIG_REMOTE_PATH' not in os.environ:
        path = '/etc/config/remote_config.json'
    else:
        path = os.environ['CONFIG_REMOTE_PATH']

    if not os.path.exists(path):
        if failing_default is not None:
            return failing_default

        while path != 'q':
            path = input("Cannot load config. Specify alternative path? [enter: q to exit]:")
            if os.path.exists(path):
                break

        if path == 'q':
            print('Exit program.')
            exit()

    with open(path, 'r') as i:
        return json.load(i)
