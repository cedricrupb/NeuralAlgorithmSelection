import taskflow as tsk
from taskflow import task_definition, backend

from tasks.train_utils import get_labels, get_preferences, get_ranking, index,\
                              get_svcomp_train_test

import os
import numpy as np
from gridfs import GridFS
import json
from tqdm import tqdm
from tasks.rank_scores import spearmann_score
from glob import iglob


def _stream_loader(base_dir):
    for path in iglob(os.path.join(base_dir, "*.json")):
        name = path[:-5]
        with open(path, "r") as i:
            D = json.load(i)
        yield name, D


def _localize_data(db, path, index, competition):

    if os.path.isdir(path):
        with open(os.path.join(path, 'index.ix'), "r") as i:
            index = json.load(i)
        return index, _stream_loader(path)

    if not os.path.exists(path):
        os.makedirs(path)

    fs = GridFS(db)
    ast_graph = db.ast_graph

    n_index = {t: i for i, t in enumerate(index)}
    o_index = []

    cur = ast_graph.find({'competition': competition})

    for obj in tqdm(cur, total=cur.count()):
        if obj['name'] in n_index:
            p = os.path.join(path, obj['name']+".json")
            if os.path.exists(p):
                continue
            fs_file = fs.get(
                obj['graph_ref']
            )
            D = fs_file.read().decode("utf-8")
            with open(p, "w") as o:
                o.write(D)
            o_index.append(obj['name'])

    with open(os.path.join(path, 'index.ix'), "w") as i:
        index = json.dump(o_index, i)

    return o_index, _stream_loader(path)


def _localize_label(db, path, index, competition, tools, category=None):

    if os.path.isfile(path):
        with open(path, "r") as i:
            return json.load(i)

    index = {t: i for i, t in enumerate(index)}

    labels = get_labels(db, competition, category, index)
    out = {}

    for category, D in labels.items():
        if category not in out:
            out[category] = {}
        for name, V in D.items():
            pref = get_preferences(V, tools)
            out[category][name] = {
                'pref': pref.tolist(),
                'pos': index[name]
            }

    with open(path, "w") as o:
        json.dump(out, o, indent=4)
    return out


def _localize_dataset(db, path, index, competition, tools, category=None):

    if not os.path.exists(path):
        os.makedirs(path)

    data_path = os.path.join(path, "data")
    label_path = os.path.join(path, "label.json")

    index, data = _localize_data(db, data_path, index, competition)
    labels = _localize_label(db, label_path, index, competition, tools,
                             category)

    return index, data, labels
