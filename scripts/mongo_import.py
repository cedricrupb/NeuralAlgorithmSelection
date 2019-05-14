from urllib.parse import quote_plus
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, BulkWriteError
import json
import gridfs
from tqdm import tqdm
import networkx as nx


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
    with open("auth.json", "r") as a:
        auth = json.load(a)

    mongodb = auth["mongodb"]
    return setup_client(mongodb["url"], mongodb["auth"])


def _get_db():
    global __db__
    if __db__ is None:
        __db__ = start_mongo()
    return __db__.graph_db


def is_forward_and_parse(e):
    if e.endswith('|>'):
        return e[:-2], True
    return e[2:], False


def parse_dfs_nx(R):
    if R is None:
        return nx.MultiDiGraph()
    graph = nx.MultiDiGraph()

    for _R in R:
        graph.add_node(_R[0], label=_R[2])
        graph.add_node(_R[1], label=_R[4])
        e_label, forward = is_forward_and_parse(_R[3])
        if forward:
            graph.add_edge(_R[0], _R[1], key=e_label)
        else:
            graph.add_edge(_R[1], _R[0], key=e_label)

    return graph


def load_graph(graph_id):
    db = _get_db()
    g = db.graphs.find_one({'label_id': graph_id})

    if g is None:
        return None

    fs = gridfs.GridFS(db)
    Js = fs.get(g['minDFS']).read().decode('utf-8')
    return json.loads(Js)


def update():
    db = _get_db()
    F = db.graphs.find(filter={'label_id': {'$exists': False}}, projection=['file'])
    for L in tqdm(F, total=F.count()):
        id = L['_id']
        if 'file' not in L:
            continue

        file = L['file']
        file = file.replace("/home/cedricr/ranking/svcomp18/sv-benchmarks/../sv-benchmarks/c/", "")
        file = file.replace(".", "_")
        file = file.replace("/", "_")
        db.graphs.update_one({'_id': id}, {'$set': {'label_id': file}})


def index():
    db = _get_db()

    index = {'counter': 0}

    F = db.wl_graphs.find(filter={'iteration': 0, 'depth': 5, 'data': {'$exists': True}}, projection=['data'])
    for L in tqdm(F, total=F.count()):

        for k in L['data'].keys():
            if k not in index:
                index[k] = index['counter']
                index['counter'] += 1

    with open("index.json", "w") as o:
        json.dump(index, o, indent=4)


def _better(A, B):
    if A['solve'] > B['solve']:
        return True
    if A['time'] >= 900:
        return False
    return A['time'] < B['time']


def ranking(D):
    count = {k: 0 for k in D.keys()}

    for i1, (tool1, K1) in enumerate(D.items()):
        for i2, (tool2, K2) in enumerate(D.items()):
            if i1 < i2:
                if _better(K1, K2):
                    count[tool1] += 1
                elif _better(K2, K1):
                    count[tool2] += 1
                else:
                    count[tool1] += 0.5
                    count[tool2] += 0.5

    count = sorted(list(count.items()), key=lambda X: X[1], reverse=True)
    out = []

    for i, (k, v) in enumerate(count):
        if i > 0:
            if v == count[i - 1][1]:
                if not isinstance(out[-1], list):
                    out[-1] = [out[-1]]
                out[-1].append(k)
            else:
                out.append(k)
        else:
            out.append(k)

    return out


def graph_labels(taskType=None):
    db = _get_db()
    filter = {}
    if not(taskType is None):
        filter['type'] = taskType
    labels = db.labels.find(filter, sort=[('graph_id', 1)])

    current_label = None
    current_type = None

    M = {
        'true': 1,
        'unknown': 0,
        'false': -1
    }

    for L in labels:
        if current_label is None:
            current_label = L['graph_id']
            current_type = L['type']
            graph_label = {}
        elif current_label != L['graph_id'] or current_type != L['type']:
            r = ranking(graph_label)
            yield (current_label, current_type, r)
            current_label = L['graph_id']
            current_type = L['type']
            graph_label = {}

        graph_label[L['tool']] = {
            'solve': M[L['solve']],
            'time': L['time']
        }


def stream_graph_label(taskType=None):
    for (graph_id, type, rank) in graph_labels(taskType):
        G = load_graph(graph_id)

        if G is None:
            print("Unknown id %s." % graph_id)
            continue

        yield (G, type, rank)


def load_wl0(graph_id):
    db = _get_db()
    cursor = db.wl_graphs.find_one(
        {
            'graph_id': graph_id,
            'iteration': 0,
            'depth': 5
        }
    )

    if cursor is None:
        return None

    return cursor['data']


def stream_wl0(taskType=None):
    for (graph_id, type, rank) in graph_labels(taskType):
        G = load_wl0(graph_id)

        if G is None:
            print("Unknown id %s." % graph_id)
            continue

        yield (G, type, rank)


def stream_wl0_to_file(taskType=None):
    data = []
    labels = []

    for (graph, type, rank) in tqdm(stream_wl0(), total=6232):
        data.append(graph)
        labels.append(rank)

    with open("data.json", "w") as o:
        json.dump(data, o, indent=4)

    with open("labels.json", "w") as o:
        json.dump(labels, o, indent=4)


def label_to_vect(label, label_index):
    vect = [0] * label_index['counter']
    vect[label_index[label]] = 1
    return vect


def graph_to_dict(G, label_index):

    node_index = {}
    nodes = []

    for i, n in enumerate(G):
        node_index[n] = i
        nodes.append(
            label_to_vect(G.nodes[n]['label'], label_index)
        )

    senders = []
    receivers = []

    # TODO Transform edges to lists


if __name__ == '__main__':
    stream_wl0_to_file()
