import numpy as np
from tqdm import tqdm
import json
import argparse
import os
import networkx as nx


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

def _simple_linear(graph, root, ast_index, ast_count, depth):
    index = {n: i for i, n in enumerate(graph.nodes())}
    ast_rev = {v: k for k, v in ast_index.items()}

    A = np.zeros((len(index), len(index)))
    D = np.zeros((len(index),))
    X = []

    for n, label in graph.nodes(data='label'):
        x = np.zeros((ast_count,))
        x[ast_index[label]] = 1
        X.append(x)

        A[index[n], index[n]] = 1

    for u, v in graph.edges():
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


def _compress_node(graph, n, ast_index, ast_count):
    ast_set = set([])

    queue = [n]

    while len(queue) > 0:
        k = queue.pop()

        for u, _, label in graph.in_edges(k, keys=True):
            if label == 's':
                ast_set.add(u)
                queue.append(u)

    label_vec = np.zeros((ast_count,))
    pos_ast = [n]
    pos_ast.extend(list(ast_set))

    for ast in pos_ast:
        node = graph.nodes[ast]
        label_vec[ast_index[node['label']]] += 1

    graph.nodes[n]['features'] = label_vec

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
        'cd': 2
    }

    for u, v, key in graph.edges(keys=True):
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
        'cd': 2
    }

    edges = []

    for u, v, key in graph.edges(keys=True):
        uix = node_index[u]
        vix = node_index[v]
        keyix = feature_index[key]
        edges.append((uix, vix, keyix))

    return {
        'nodes': node_array,
        'edges': edges
    }


def load_labels(path):

    with open(path, "r") as i:
        label_load = json.load(i)

    matrix = label_load['matrix']
    index = label_load['index']['unreach-call']

    labels = {}

    for k, i in index.items():
        labels[k] = matrix[i][0]

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()

    labels = load_labels(args.label)

    ast_index = {}
    ast_count = 158

    try:
        for file, label in tqdm(labels.items()):
            identifier = file.replace('.', '_').replace("/", "_")

            in_file = os.path.join(args.input, identifier+'.dfs')
            out_file = os.path.join(args.output, identifier+'.json')

            if os.path.exists(out_file):
                continue

            if not os.path.exists(in_file):
                print("Unknown path %s." % in_file)
                continue

            with open(in_file, "r") as i:
                G = json.load(i)

            G = parse_dfs_nx(G)
            compress_graph(G, ast_index, ast_count)

            with open(out_file, "w") as o:
                json.dump(
                    to_custom_dict(G), o
                )

    finally:
        with open("ast_index.json", "w") as o:
            json.dump(ast_index, o, indent=4)
