from taskflow import config as cfg
from urllib.parse import quote_plus
from pymongo import MongoClient

import argparse
from gridfs import GridFS
from tqdm import tqdm
import json

__config__ = None

__db__ = None
__client__ = {}


def setup_client(url, auth=None):
    global __client__
    if url not in __client__:
        if auth is not None:
            uri = 'mongodb://%s:%s@%s/%s' % (
                quote_plus(auth['username']),
                quote_plus(auth['password']),
                url,
                auth['authSource']
            )
        else:
            uri = 'mongodb://%s/' % url

        __client__[url] = MongoClient(uri)
    return __client__[url]


def start_mongo():
    config = get_config()

    if len(config) == 0:
        return None

    if 'execution' not in config:
        return None

    auth = config['execution']
    mongodb = auth["mongodb"]
    return setup_client(mongodb["url"], mongodb["auth"])


def get_db():
    global __db__
    if __db__ is None:
        __db__ = start_mongo()

    config = get_config()
    if 'execution' not in config:
        return None

    return __db__[__config__['execution']['mongodb']['database']]


def get_config():
    global __config__

    if __config__ is None:
        __config__ = cfg.load_config(failing_default={})

    return __config__


db = get_db()
graph = db.svcomp
stat = db.graph_statistics

sizes = {}
cursor = stat.find({'competition': '2018'}, ['name', 'cfg_edges', 'pdg_edges'])

for obj in tqdm(cursor, total=cursor.count()):
    sizes[obj['name']] = obj['cfg_edges'] + obj['pdg_edges']


filter = {
    'svcomp': '2018'
}

times = {}
cursor = graph.find(filter, ['name', 'file'])

for obj in tqdm(cursor, total=cursor.count()):
    name = obj['file'].replace("../sv-benchmarks/c/", "")
    if obj['name'] in sizes and name not in filter:
        times[name] = sizes[obj['name']]

with open("conv_size.json", "w") as o:
    json.dump(times, o, indent=4)
