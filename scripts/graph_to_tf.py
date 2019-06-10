import tensorflow as tf
import argparse

from tqdm import tqdm
from glob import glob
import json

import numpy as np


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(D):

    nodes = []
    edges = []
    sender = []
    receiver = []

    for features in D['nodes']:
        F = [0] * 148
        for p, v in features:
            F[p] = v
        nodes.append(F)

    for u, v, e in D['edges']:
        E = [0]*3
        E[e] = 1
        edges.append(E)
        sender.append(u)
        receiver.append(v)

    features = {
        'nodes': _float_feature(np.array(nodes, dtype=np.float32)),
        'edges': _int64_feature(np.array(edges)),
        'sender': _int64_feature(np.array(sender)),
        'receiver': _int64_feature(np.array(receiver))
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("output_dir")

args = parser.parse_args()

for f in glob(args.input_dir + "*.json"):

    with open(f, "r") as i:
        D = json.load(i)

    example = create_example(D)
