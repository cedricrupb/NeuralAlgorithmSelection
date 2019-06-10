import taskflow as tsk
from taskflow import task_definition, backend

from tasks import train_utils

import tensorflow as tf

import os
from tqdm import tqdm
from gridfs import GridFS
import json
import traceback
import numpy as np
import gzip


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def seq_feat(dtype):
    return tf.FixedLenSequenceFeature([1], dtype, allow_missing=True)


graph_description = {
    'globals': tf.FixedLenFeature([], tf.int64),
    'nodes_index_0': seq_feat(tf.int64),
    'nodes_index_1': seq_feat(tf.int64),
    'nodes_content': seq_feat(tf.float32),
    'edges': seq_feat(tf.int64),
    'sender': seq_feat(tf.int64),
    'receiver': seq_feat(tf.int64),
    'preference': seq_feat(tf.float32)
}


def get_example(string):
    return tf.parse_single_example(string, graph_description)


def create_example(D, cat, pref):

    nodes = []
    nodes_ix0 = []
    nodes_ix1 = []
    edges = []
    sender = []
    receiver = []

    for i, features in enumerate(D['nodes']):
        for p, v in features:
            nodes_ix1.append(p)
            nodes_ix0.append(i)
            nodes.append(v)

    for u, v, e in D['edges']:
        edges.append(e)
        sender.append(u)
        receiver.append(v)

    categ = {
        'reachability': 0,
        'termination': 1,
        'memory': 2,
        'overflow': 3
    }

    features = {
        'globals': _int64_feature([categ[cat]]),
        'nodes_index_0': _int64_feature(nodes_ix0),
        'nodes_index_1': _int64_feature(nodes_ix1),
        'nodes_content': _float_feature(nodes),
        'edges': _int64_feature(edges),
        'sender': _int64_feature(sender),
        'receiver': _int64_feature(receiver),
        'preference': _float_feature(pref)
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def to_gz(path):
    with open(path, "rb") as i:
        with open(path+".gz", "wb") as o:
            o.write(
                gzip.compress(
                    i.read(), compresslevel=5
                )
            )
    os.remove(path)
    return path+".gz"


@task_definition()
def download(tools, competition, train, test, category=None, env=None):

    db = env.get_db()

    train = set(train)
    test = set(test)

    graph = db.ast_graph

    cur = graph.find({'competition': competition})

    fs = GridFS(db)
    cat_name = category
    if cat_name is None:
        cat_name = 'all'
    train_dir = os.path.join(env.get_cache_dir(), 'train_%s_%s.tfr' % (competition, cat_name))
    test_dir = os.path.join(env.get_cache_dir(), 'test_%s_%s.tfr' % (competition, cat_name))

    labels = train_utils.get_labels(db, competition, category=category)

    with tf.python_io.TFRecordWriter(train_dir) as train_writer:
        with tf.python_io.TFRecordWriter(test_dir) as test_writer:
            for obj in tqdm(cur, total=cur.count()):

                train_t = obj['name'] in train
                test_t = obj['name'] in test

                if not train_t and not test_t:
                    continue

                try:
                    file = fs.get(obj['graph_ref']).read().decode('utf-8')
                    file = json.loads(file)

                    for cat, lookup in labels.items():
                        if obj['name'] in lookup:
                            pref = lookup[obj['name']]
                            pref = train_utils.get_preferences(pref, tools)
                            example = create_example(file, cat, pref)
                            if train_t:
                                train_writer.write(example.SerializeToString())
                            else:
                                test_writer.write(example.SerializeToString())
                except Exception:
                    traceback.print_exc()
                    continue

    return to_gz(train_dir), to_gz(test_dir)


if __name__ == '__main__':

    dataset_key = 'initial_test'
    lr_key = "initial_reachability_0"
    competition = "2019"
    category = "reachability"

    tools, train_index, test_index = train_utils.get_svcomp_train_test(
        dataset_key, competition, category, test_ratio=0.2
    )

    down = download(tools, competition, train_index, test_index, category)

    with backend.openLocalSession() as sess:
        sess.run(
            down
        )
