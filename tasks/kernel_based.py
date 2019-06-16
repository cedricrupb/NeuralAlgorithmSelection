import taskflow as tsk
from taskflow import task_definition, backend

from gridfs import GridFS
import json
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import trange
from time import time


def _localize_wl(fs, wl_col, id, iteration, depth, index):
    wl_item = wl_col.find_one({'_id': id, 'max_depth': depth})

    if wl_item is None:
        raise ValueError("Unknown id %s for depth %d." % (str(id), depth))

    wl_refs = wl_item['wl_refs']
    if iteration >= len(wl_refs):
        raise ValueError("ID %s exists not for iteration %d" % (str(id), iteration))

    D = json.loads(fs.get(wl_refs[iteration]).read().decode('utf-8'))
    R = {}

    for p, v in D.items():
        if p not in index:
            index[p] = index['count']
            index['count'] += 1
        R[index[p]] = v
    return R


def _bag_to_vec(bag, n):
    vec = np.zeros(n, dtype=np.uint64)

    for p, v in bag.items():
        vec[p] = v

    return vec


def _localize_wls(db, ids, iteration, depth, ast_index):
    wl = db.wl_features
    fs = GridFS(db)

    result = {}
    for id in ids:
        result[id] = _localize_wl(fs, wl, id, iteration,
                                  depth, ast_index)

    for id in ids:
        D = _bag_to_vec(result[id], ast_index['count'])
        result[id] = D

    return result


@task_definition()
def load_wl_ids(competition, env=None):

    if env is None:
        raise ValueError("train_test_split needs an execution context")

    db = env.get_db()
    wl = db.wl_features

    wls = wl.find({'competition': competition}, ['_id‚'])
    return [w['_id'] for w in wls]


@task_definition()
def dual_scatter(iterable, n):
    sc = [[] for _ in range(n)]

    for i, value in enumerate(iterable):
        sc[i % n] = value

    dual = []
    for i in range(n):
        for j in range(i, n):
            dual.append([sc[i], sc[j]])

    return dual


def to_matrix(ids, wls):
    r = [wls[id] for id in ids]
    r = np.vstack(r)
    return r


def _jaccard_between(X, Y):

    min_sum = np.minimum(X, Y).sum(axis=1, dtype=np.float64)
    max_sum = np.maximum(X, Y).sum(axis=1, dtype=np.float64)

    return min_sum / max_sum


def jaccard_kernel(X):
    row = list(range(X.shape[0]))
    columns = list(range(X.shape[0]))
    data = [1] * X.shape[0]

    for i in trange(1, X.shape[0]):
        X_index = np.arange(i, X.shape[0])
        Y_index = np.arange(0, X.shape[0] - i)
        X_slice = X[X_index, :]
        Y_slice = X[Y_index, :]
        S = _jaccard_between(X_slice, Y_slice)
        S = list(S.flat)
        row.extend(X_index.data)
        columns.extend(Y_index.data)
        data.extend(S)
        row.extend(Y_index.data)
        columns.extend(X_index.data)
        data.extend(S)

    return coo_matrix((data, (row, columns))).toarray()


def jaccard_pair(X, Y):
    stack = []

    if X.shape[0] < Y.shape[0]:
        return jaccard_pair(Y, X).transpose()

    for i in trange(Y.shape[0]):
        yix = [i]*X.shape[0]
        Yix = Y[yix, :]
        B = _jaccard_between(X, Yix)
        B = np.reshape(B, [B.shape[0], 1])
        stack.append(B)

    return np.hstack(stack)


def single_kernel(wls, ids):
    X = to_matrix(ids, wls)
    return jaccard_kernel(X)


def pair_kernel(wls, row_ids, col_ids):
    X = to_matrix(row_ids, wls)
    Y = to_matrix(col_ids, wls)
    return jaccard_pair(X, Y)


def load_kernel(db, _id):
    kernel = db.kernels

    f = kernel.find_one({'_id': _id}, ['kernel_ref', 'transpose'])
    if f is None:
        raise ValueError("Unknown id %s." % (str(_id)))

    fs = GridFS(db)
    transpose = f['transpose']
    obj_id = f['kernel_ref']

    K = np.array(
        json.loads(
            fs.get(obj_id).read().decode('utf-8')
        )
    )

    if transpose:
        K = K.transpose()
    return K


@task_definition()
def wl_kernel(id, iteration, depth, row_ids, col_ids=None,
              dual=False, env=None):
    if env is None:
        raise ValueError("train_test_split needs an execution context")

    if col_ids is None:
        col_ids = row_ids

    db = env.get_db()
    kernel = db.kernels

    if dual:
        f1 = kernel.find_one({'kernel_id': id})
        f2 = kernel.find_one({'kernel_id': id+"_t"})
        if f1 is not None and f2 is not None:
            return f1['_id'], f2['_id']
    else:
        f = kernel.find_one({'kernel_id': id})

        if f is not None:
            return f['_id']

    ast_index = {'count': 0}

    ids = set.union(set(row_ids), set(col_ids))
    corner = len(ids) > len(row_ids) or len(ids) > len(col_ids)
    wls = _localize_wls(db, ids, iteration, depth, ast_index)

    start = time()
    if corner:
        K = pair_kernel(wls, row_ids, col_ids)
    else:
        K = single_kernel(wls, row_ids)
    run_time = time() - start

    fs = GridFS(db)
    fid = fs.put(
        json.dumps(K.tolist()).encode('utf-8'),
        iteration=iteration, depth=depth, kernel_id=id
    )

    id1 = kernel.insert_one({
        'kernel_id': id,
        'transpose': False,
        'kernel_ref': fid,
        'iteration': iteration,
        'depth': depth,
        'row_ids': row_ids,
        'col_ids': col_ids,
        'run_time': run_time
    })

    if dual:
        id2 = kernel.insert_one({
            'kernel_id': id+"_t",
            'transpose': True,
            'kernel_ref': fid,
            'iteration': iteration,
            'depth': depth,
            'row_ids': col_ids,
            'col_ids': row_ids,
            'run_time': run_time
        })
        return id1, id2
    else:
        return id1


@task_definition()
def merge_wl_kernel(ids, dual=False, env=None):
    pass


if __name__ == '__main__':
    ids = load_wl_ids('2019')
    ids1 = ids[:50]
    ids2 = ids[50:75]
    kernel = wl_kernel('test_kernel', 0, 5, ids1, ids2)
    with backend.openLocalSession(auto_join=True) as sess:
        sess.run(kernel)
