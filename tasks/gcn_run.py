import taskflow as tsk
from taskflow import task_definition, backend

from tasks import train_utils, gcn_model

import os
from gridfs import GridFS
import json
from random import shuffle
import copy
import traceback

import numpy as np
import tensorflow as tf

from tasks.download_to_tf import get_example


class GraphLoader(object):

    def __init__(self, db, path, competition, n_features=148):
        self._db = db
        self._path = path
        self._comp = competition
        self._fs = GridFS(self._db)
        self._feat = n_features
        self._buffer = {}

    def _load_to_disk(self, name, path):
        ast_graph = self._db.ast_graph

        info = ast_graph.find_one({'name': name, 'competition': self._comp})

        if info is None:
            raise ValueError("Unknown graph %s [svcomp: %s]" % (name, self._comp))

        graph_ref = info['graph_ref']
        D = self._fs.get(graph_ref).read().decode('utf-8')

        with open(path, "w") as o:
            o.write(D)

        return json.loads(D)

    def _graph_to_dict(self, D):

        nodes = []
        edges = []
        sender = []
        receiver = []

        for features in D['nodes']:
            F = [0] * self._feat
            for p, v in features:
                F[p] = v
            nodes.append(F)

        for u, v, e in D['edges']:
            E = [0]*3
            E[e] = 1
            edges.append(E)
            sender.append(u)
            receiver.append(v)

        return {
            'nodes': nodes,
            'edges': edges,
            'senders': sender,
            'receivers': receiver
        }

    def _load(self, name):
        path = os.path.join(self._path, self._comp)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, name+".json")

        print("Load graph %s [svcomp: %s]" % (name, self._comp))

        if not os.path.isfile(path):
            D = self._load_to_disk(name, path)
        else:
            with open(path, "r") as i:
                D = json.load(i)

        return self._graph_to_dict(D)

    def load(self, name):
        if name not in self._buffer:
            self._buffer[name] = self._load(name)
        return self._buffer[name]


class GraphLabelLoader(object):

    def __init__(self, db, tools, competition):
        self._db = db
        self._comp = competition
        self._buffer = {}
        self._tools = tools

    def _load(self, name, category):
        label = train_utils.get_single_label(
            self._db, name, self._comp, category
        )

        if len(label) == 0:
            raise ValueError("Unknown graph %s [svcomp: %s, category: %s]"
                             % (name, self._comp, category))

        pref = {
            cat: train_utils.get_preferences(
                l, self._tools
            )
            for cat, l in label.items()
        }

        return pref

    def load(self, name, category):
        if category not in self._buffer:
            self._buffer[category] = {}
        cat = self._buffer[category]

        if name not in cat:
            cat[name] = self._load(name, category)
        return cat[name]


def build_gt_generator(db, tools, competition, index, cache_path,
                       category=None, batch_size=32, to_shuffle=True):

    gLoader = GraphLoader(db, cache_path, competition)
    lLoader = GraphLabelLoader(db, tools, competition)

    cat = {
        'reachability': 0,
        'termination': 1,
        'memory': 2,
        'overflow': 3
    }

    def gen():
        if to_shuffle:
            shuffle(index)
        it = int(len(index)/batch_size) + 1

        for i in range(it):
            lower = i * batch_size
            upper = min((i + 1) * batch_size - 1, len(index))
            batch = index[lower:upper]

            if len(batch) == 0:
                continue

            graphs = []
            labels = []
            for name in batch:
                try:
                    g = gLoader.load(name)
                    label = lLoader.load(name, category)

                    for c, L in label.items():
                        globals = [0]*4
                        globals[cat[c]] = 1
                        g = copy.copy(g)
                        g['globals'] = globals
                        graphs.append(g)
                        labels.append(L)
                except ValueError:
                    traceback.print_exc()
                    continue
            labels = np.vstack(labels)
            yield graphs, labels

    return gen


def build_full(db, tools, competition, index, cache_path,
               category=None):

    gLoader = GraphLoader(db, cache_path, competition)
    lLoader = GraphLabelLoader(db, tools, competition)

    cat = {
        'reachability': 0,
        'termination': 1,
        'memory': 2,
        'overflow': 3
    }

    graphs = []
    labels = []

    for name in index:
        try:
            g = gLoader.load(name)
            label = lLoader.load(name, category)

            for c, L in label.items():
                globals = [0]*4
                globals[cat[c]] = 1
                g = copy.copy(g)
                g['globals'] = globals
                graphs.append(g)
                labels.append(L)
        except ValueError:
            traceback.print_exc()
            continue

    labels = np.vstack(labels)
    return graphs, labels


def repeated_gen(generator, epochs):
    for _ in range(epochs):
        for i in generator():
            yield i
        yield 'STOP'


@task_definition()
def gcn_train_test(key, model_key, tools, train_index, test_index, competition,
                   category=None,
                   batch_size=32, epoch=1, model_args={},
                   env=None):
    if env is None:
        raise ValueError("train_test_split needs an execution context")
    db = env.get_db()

    shuffle(train_index)
    validate = train_index[:int(len(train_index)*0.1)]
    train = train_index[int(len(train_index)*0.1):]

    train = build_gt_generator(
        db, tools, competition, train,
        env.get_cache_dir(), category,
        batch_size=batch_size
    )
    train = repeated_gen(train, epoch)

    # validate = build_full(
    #    db, tools, competition, validate,
    #    env.get_cache_dir(), category
    # )

    model = gcn_model.train_model(
        model_key, train, validate, env=env
    )


def setup_dataset(tfr_path):
    dataset = tf.data.TFRecordDataset(tfr_path, compression_type="GZIP")
    dataset = dataset.map(get_example)

    return dataset


@task_definition()
def gcn_tfr_train(key, model_key, tools, train, test, competition,
                  category=None,
                  batch_size=32, epoch=1, model_args={},
                  env=None):
    if env is None:
        raise ValueError("train_test_split needs an execution context")

    train = os.path.join(env.get_cache_dir(), train)
    test = os.path.join(env.get_cache_dir(), test)

    train = setup_dataset(train)
    test = setup_dataset(test)

    val_count = 200
    # train = train.shuffle(buffer_size=128)
    # validate = train.take(val_count)
    # train = train.skip(val_count)
    # train = train.prefetch(batch_size).batch(batch_size).repeat(epoch)

    model = gcn_model.train_test_dataset(
        model_key, train, None, test, env=env
    )


if __name__ == '__main__':
    dataset_key = 'initial_test'
    lr_key = "initial_reachability_0"
    competition = "2019"
    category = "reachability"

    tools, train_index, test_index = train_utils.get_svcomp_train_test(
        dataset_key, competition, category, test_ratio=0.2
    )

    lr = gcn_tfr_train(
        lr_key, "", tools,
        'train_2019_reachability.tfr.gz',
        'test_2019_reachability.tfr.gz',
        competition, category
    )

    with backend.openLocalSession() as sess:
        sess.run(lr)
