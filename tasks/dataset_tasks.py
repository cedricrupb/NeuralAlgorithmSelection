import taskflow as tsk
from taskflow.task import task_definition
from taskflow.backend import openLocalSession

import requests
import math
from tqdm import tqdm
import os
import bz2

from zipfile import ZipFile
from glob import glob
import xml.etree.ElementTree as ET

import hashlib
from pymongo import UpdateOne
from bson.objectid import ObjectId

import subprocess as sp
from gridfs import GridFS
import shutil

import networkx as nx
import numpy as np
import json
import random


types_str = {
    'memory': ['valid-deref',
               'valid-free',
               'valid-memtrack',
               'valid-memcleanup',
               'valid-memsafety'],
    'reachability': ['unreach-call'],
    'overflow': ['no-overflow'],
    'termination': ['termination']
}

SV_BENCHMARKS_GIT = "https://github.com/sosy-lab/sv-benchmarks.git"


def _download_file(url, path):
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0

    print("Download url: %s >> %s" % (url, path))
    with open(path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size/block_size), unit='KB'):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size and os.path.getsize(path) != total_size:
        os.remove(file)
        raise ValueError("ERROR, something went wrong")


def create_folder(path):
    if len(os.path.splitext(path)[1]) > 0:
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        os.makedirs(path)


def read_svcomp_data(path):
    D = {}

    directory = os.path.dirname(path)
    tmp_dir = os.path.join(directory, "tmp")

    try:
        with ZipFile(path, 'r') as zipObj:
            zipObj.extractall(path=tmp_dir)

        for entry in tqdm(glob(os.path.join(tmp_dir, "*.bz2"))):
            read_svcomp_bz2(entry, D)

    finally:
        if os.path.exists(tmp_dir):
            os.system('rm -rf %s' % tmp_dir)

    return D


def detect_type(text):
    global types_str

    for k, V in types_str.items():
        for v in V:
            if v in text:
                return k

    raise ValueError("Unknown type for properties \"%s\"" % text)


def short_name(name):
    name = name.replace("../sv-benchmarks/c/", "")
    name = name.replace("/", "_")
    name = name.replace(".", "_")
    return name


def ground_truth(file, type):
    global types_str

    for type_str in types_str[type]:
        if 'true-%s' % type_str in file:
            return True
        if 'false-%s' % type_str in file:
            return False

    raise ValueError("Cannot detect type %s in file \"%s\"." % (type, file))


def read_svcomp_bz2(path, result):
    with bz2.open(path) as o:
        xml = o.read()

    root = ET.fromstring(xml)

    if 'java' in root.attrib['name'].lower():
        return

    tool_name = root.attrib['benchmarkname']

    for run in root.iter('run'):
        attr = run.attrib
        file = attr['name']
        name = short_name(file)
        category = detect_type(attr['properties'])

        for column in run:
            title = column.attrib['title']

            if title == 'status':
                status = column.attrib['value']

            if title == 'cputime':
                cputime = float(column.attrib['value'][:-1])

        if category not in result:
            result[category] = {}
        if tool_name not in result[category]:
            result[category][tool_name] = {}
        result[category][tool_name][name] = {
            'file': file,
            'status': status,
            'ground_truth': ground_truth(file, category),
            'cputime': cputime
        }


def sp_run_success(cmd):
    sp.run(cmd, check=True, shell=True)


def prepare_svcomp_git(competition_year, directory):

    base_path = os.path.join(directory, "svcomp-git")
    git_path = os.path.join(base_path, "sv-benchmarks")

    create_folder(base_path)

    if not os.path.exists(git_path):
        global SV_BENCHMARKS_GIT
        sp_run_success(["cd", "%s" % base_path, "&&", "git",
                        "clone", SV_BENCHMARKS_GIT])

    if not os.path.exists(git_path):
        raise tsk.EnvironmentException(
                    "Something went wrong during git clone.")

    sp_run_success(["cd", "%s" % git_path, "&&", "git", "checkout",
                    "tags/svcomp%s" % competition_year])

    return git_path


def run_pesco(pesco_path, in_file, out_file, heap="10g", timeout=900):
    path_to_source = in_file

    run_path = os.path.join(pesco_path, "scripts", "cpa.sh")
    output_path = out_file

    if not os.path.isdir(pesco_path):
        raise ValueError("Unknown pesco path %s" % pesco_path)

    if not (os.path.isfile(path_to_source) and (path_to_source.endswith('.i') or path_to_source.endswith('.c'))):
        raise ValueError('path_to_source is no valid filepath. [%s]' % path_to_source)

    proc = sp.run(
                    [run_path,
                     "-graphgen",
                     "-heap", heap,
                     "-Xss512k",
                     "-setprop", "neuralGraphGen.output="+output_path,
                     path_to_source
                     ],
                    check=True,
                    stdout=sp.PIPE,
                    stderr=sp.PIPE,
                    timeout=timeout
                    )


# Tasks

@task_definition()
def load_svcomp(competition_year, env=None):
    url = "https://sv-comp.sosy-lab.org/%s/results/results-verified/All-Raw.zip" % competition_year

    path = "./svcomp/svcomp_%s.zip" % competition_year

    if not env or not env.is_remote_io_loaded():
        raise tsk.EnvironmentException("Need a remote context to process competition data")

    coll = env.get_db().svcomp

    if coll.find_one({'svcomp': competition_year}) is not None:
        return [
            r['_id'] for r in coll.find({'svcomp': competition_year}, ['_id'])
        ]

    tmp = env.get_cache_dir()
    if tmp is not None:
        path = os.path.join(tmp, path)

    create_folder(path)

    if not os.path.exists(path):
        _download_file(url, path)

    if not os.path.exists(path):
        raise ValueError("Something went wrong for competition: %s" % competition_year)

    comp = read_svcomp_data(path)

    updates = []
    names = set([])
    for category, V in comp.items():

        for tool, D in V.items():

            for name, entry in D.items():

                update = {
                    'name': name,
                    'svcomp': competition_year,
                    'category': category,
                    'tool': tool
                }
                update.update(entry)

                identifier = '::'.join(sorted(['%s_%s' % (k, v) for k, v in update.items()]))
                identifier = hashlib.blake2b(identifier.encode('utf-8'), digest_size=12).digest()
                identifier = ObjectId(identifier)
                update['_id'] = identifier
                names.add(update['name'])

                updates.append(UpdateOne({
                    '_id': identifier
                }, {'$set': update}, upsert=True))

    coll.bulk_write(updates)

    return list(names)


@task_definition(workload="heavy")
def code_to_graph(name, competition, env=None):
    if 'PESCO_PATH' not in os.environ:
        raise tsk.EnvironmentException(
                    "Environment variable PESCO_PATH has to be defined!")

    if not env or not env.is_remote_io_loaded():
        raise tsk.EnvironmentException(
                    "Need a remote context to process competition data")

    svcomp_db = env.get_db().svcomp

    info = svcomp_db.find_one({'name': name, 'svcomp': competition})

    if info is None:
        raise ValueError("Unknown task id %s [SVCOMP %s]" % (str(name), competition))

    if 'graph_ref' in info:
        return info['graph_ref']

    svcomp_db.update({'name': info['name'], 'svcomp': info['svcomp']},
                     {'$set': {'graph_ref': 0}})

    try:

        git_path = prepare_svcomp_git(info['svcomp'], env.get_host_cache_dir())
        file_path = info['file'].replace("../sv-benchmarks/", "")
        file_path = os.path.join(git_path, file_path)

        if not os.path.isfile(file_path):
            raise ValueError("Some problem occur while accessing file %s." % file_path)

        pesco_path = os.environ['PESCO_PATH']
        out_path = info['name'] + ".json"
        out_path = os.path.join(env.get_cache_dir(), out_path)

        run_pesco(
            pesco_path,
            file_path,
            out_path
        )

        if not os.path.exists(out_path):
            raise tsk.EnvironmentException(
                "Pesco doesn't seem to be correctly configured! No output for %s" % info['name']
            )

        fs = GridFS(env.get_db())
        file = fs.new_file(name=info['name'], competition=info['svcomp'], encoding="utf-8")

        try:
            with open(out_path, "r") as i:
                shutil.copyfileobj(i, file)
        finally:
            file.close()

        svcomp_db.update({'name': info['name'], 'svcomp': info['svcomp']},
                         {'$set': {'graph_ref': file._id}})

    except Exception as e:
        svcomp_db.update({'name': info['name'], 'svcomp': info['svcomp']},
                         {'$unset': {'graph_ref': 0}})
        raise e

    return file._id


def is_forward_and_parse(e):
    if e.endswith('|>'):
        return e[:-2], True
    return e[2:], False


def parse_dfs_nx_alt(R):
    if R is None:
        return nx.MultiDiGraph()
    graph = nx.MultiDiGraph()

    for k, v in R['nodes'].items():
        graph.add_node(k, label=v)

    for u, l, v in R['edges']:
        e_label, forward = is_forward_and_parse(l)
        if forward:
            graph.add_edge(u, v, key=e_label)
        else:
            graph.add_edge(v, u, key=e_label)

    for n in graph.nodes():
        if graph.out_degree(n) == 0:
            s = n.split("_")
            if len(s) == 2:
                graph.add_edge(n, s[0], key="s")

    return graph


def load_graph(task_id, env):
    if not env or not env.is_remote_io_loaded():
        raise tsk.EnvironmentException(
                    "Need a remote context to process competition data")

    svcomp_db = env.get_db().svcomp

    info = svcomp_db.find_one({'_id': task_id})

    if info is None:
        raise ValueError("Unknown task id %s" % str(task_id))

    if 'graph_def' not in info:
        raise ValueError("Graph %s is not computed" % str(info['name']))

    fs = GridFS(env.get_db())
    text = fs.get(info['graph_ref']).decode('utf-8')

    return parse_dfs_nx_alt(json.loads(text))


def subsample_graph(graph, root, size):

    if size > len(graph.nodes()):
        return list(graph.nodes())

    # print("Has to subsample as %i > %i (%s)" % (len(graph.nodes()), size, str(graph.nodes[root]['label'])))

    depths = {}
    seen = set([])

    queue = [(root, 0)]

    while len(queue) > 0:
        node, depth = queue.pop()

        if node in seen:
            continue

        if depth not in depths:
            depths[depth] = []

        depths[depth].append(node)
        seen.add(node)

        for v, _ in graph.in_edges(node):
            queue.append((v, depth + 1))

    sample = []

    i = 0
    while i < len(depths) and len(sample) + len(depths[i]) <= size:
        sample.extend(depths[i])
        i += 1

    if i < len(depths) and size - len(sample) > 0:
        sample.extend(random.sample(depths[i], size - len(sample)))

    return sample


def _simple_linear(graph, root, ast_index, ast_count, depth, sub_sample=50, verbose=False):
    index = {n: i for i, n in enumerate(subsample_graph(graph, root, sub_sample))}

    A = np.zeros((len(index), len(index)))
    D = np.zeros((len(index),))
    X = []

    for n in index.keys():
        label = graph.nodes[n]['label']
        x = np.zeros((ast_count,))
        x[ast_index[label]] = 1
        X.append(x)

        A[index[n], index[n]] = 1

    for u, v in graph.edges(index.keys()):
        if v not in index:
            continue
        A[index[u], index[v]] = 1
        A[index[v], index[u]] = 1

    D = np.sum(A, axis=1)
    D = np.diag(1/np.sqrt(D))
    S = D.dot(A).dot(D)

    X = np.vstack(X)

    S = np.linalg.matrix_power(S, depth)
    X = S.dot(X)

    return X[index[root], :]


def _compress_node2(graph, n, ast_index, ast_count):
    ast_set = set([])

    queue = [n]

    while len(queue) > 0:
        k = queue.pop()

        ast_set.add(k)

        for u, _, label in graph.in_edges(k, keys=True):
            if label == 's':
                queue.append(u)

    G_ast = graph.subgraph(ast_set)
    vec = _simple_linear(G_ast, n, ast_index, ast_count, 6)

    graph.nodes[n]['features'] = vec

    ast_set.remove(n)

    return ast_set


def compress_graph(graph, ast_index, ast_count):
    if 'count' not in ast_index:
        ast_index['count'] = 0

    for n in graph.nodes():
        node = graph.nodes[n]
        label = node['label']
        if label not in ast_index:
            ast_index[label] = ast_index['count']
            ast_index['count'] += 1

    if ast_index['count'] > ast_count:
        raise ValueError("Detect at least %i AST labels" % ast_index['count'])

    cfg_nodes = set([])

    for u, v, label in graph.edges(keys=True):
        if label == 'cfg':
            cfg_nodes.add(u)
            cfg_nodes.add(v)

    ast = set([])
    for cfg_node in cfg_nodes:
        ast = ast.union(
            _compress_node2(graph, cfg_node, ast_index, ast_count)
        )

    graph.remove_nodes_from(ast)

    feature_index = {
        'cfg': 0,
        'dd': 1,
        'cd': 2,
        'cd_f': 2,
        'cd_t': 2
    }

    for u, v, key in graph.edges(keys=True):
        if key == 'du':
            continue
        feature = np.zeros((3,))
        feature[feature_index[key]] = 1
        graph.edges[u, v, key]['features'] = feature


def to_custom_dict(graph):
    count = 0
    node_index = {}

    node_array = []

    for n, features in graph.nodes(data='features'):
        node_index[n] = count
        count += 1

        if features is None:
            continue

        node_embed = []
        for i in range(features.shape[0]):
            val = features[i]
            if val > 0:
                node_embed.append((i, val))
        node_array.append(node_embed)

    # node_array = np.vstack(node_array)

    feature_index = {
        'cfg': 0,
        'dd': 1,
        'cd': 2,
        'cd_f': 2,
        'cd_t': 2
    }

    edges = []

    for u, v, key in graph.edges(keys=True):
        if key == 'du':
            continue
        uix = node_index[u]
        vix = node_index[v]
        keyix = feature_index[key]
        edges.append((uix, vix, keyix))

    return {
        'nodes': node_array,
        'edges': edges
    }


@task_definition(timeout=600)
def ast_features_graph(task_id, sub_sample, env=None):
    graph = load_graph(task_id, env)

    db = env.get_db()
    cache = db.cache

    ast_index = {}
    ast_count = 158
    p = cache.find_one({'_id': 'ast_index'})

    if p is not None:
        ast_index = p['value']

    try:

        compress_graph(graph, ast_index, ast_count)

        graph_repr = json.dumps(
            to_custom_dict(graph)
        )

        fs = GridFS(env.get_db())

    finally:
        cache.update_one({'_id': 'ast_index'},
                         {'$set': {'value': ast_index}},
                         upsert=True)


@task_definition(timeout=600)
def ast_features_bag(task_id, depth, env=None):
    pass


if __name__ == '__main__':

    comp = "2019"

    load = load_svcomp(comp)
    load_it = tsk.fork(load)
    graph_it = code_to_graph(load_it, comp)

    with openLocalSession() as sess:
        sess.run(graph_it)
