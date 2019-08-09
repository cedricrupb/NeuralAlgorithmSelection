import taskflow as tsk
from taskflow import task_definition, backend
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

import numpy as np


def build_filter_func(filter):

    if filter is not None:
        filter = set(filter)

        def filter_func(X):
            return X in filter
    else:
        def filter_func(X):
            return True

    return filter_func


def retrieve_benchmark_names(db, competition, category=None, filter=None):

    search = {
        'svcomp': competition,
        'graph_ref': {'$exists': 1}
    }
    if category is not None:
        search['category'] = category

    pipeline = [
        {'$match': search},
        {
            '$group': {
                '_id': "$name"
            }
        }
    ]

    filter = build_filter_func(filter)

    names = []

    for name_obj in db.svcomp.aggregate(pipeline):
        if filter(name_obj['_id']):
            names.append(name_obj['_id'])

    return names


@task_definition()
def train_test(key, competition, test_ratio=0.1,
               category=None, filter=None, env=None):

    if env is None:
        raise ValueError("train_test_split needs an execution context")

    db = env.get_db()
    data_split = db.data_split

    f = data_split.find_one({'key': key, 'competition': competition})

    if f is not None:
        return f['train'], f['test']

    names = retrieve_benchmark_names(
        db, competition, category=category, filter=filter
    )

    name_train, name_test = train_test_split(
        names, test_size=test_ratio, random_state=42
    )

    data_split.insert({
        'key': key, 'competition': competition, 'category': category,
        'test_ratio': test_ratio, 'train': name_train, 'test': name_test,
        'train_size': len(name_train), 'test_size': len(name_test)
    })

    return name_train, name_test


@task_definition()
def train_crossval(key, competition, cv=10,
                   category=None, filter=None, env=None):

    if env is None:
        raise ValueError("train_test_split needs an execution context")

    db = env.get_db()
    data_split = db.data_split

    f = data_split.find({'competition': competition,
                         'crossval': key})

    if f.count() == cv:
        return [[a['key'], a['train'], a['test']] for a in f]

    names = retrieve_benchmark_names(
        db, competition, category=category, filter=filter
    )

    names = np.array(names)

    kf = KFold(n_splits=cv, shuffle=True)

    name_split = [
        ["%s_%i" % (key, i), names[train].tolist(), names[test].tolist()]
        for i, (train, test) in enumerate(kf.split(names))
    ]

    for id, names_train, names_test in name_split:
        test_ratio = len(names_test) / (len(names_train) + len(names_test))
        data_split.insert({
            'key': id, 'competition': competition,
            'category': category,
            'test_ratio': test_ratio, 'train': names_train, 'test': names_test,
            'train_size': len(names_train), 'test_size': len(names_test),
            'crossval': key
        })

    return name_split


@task_definition()
def tool_coverage(competition, filter=None, category=None, env=None,
                  cache_key=None):

    if env is None:
        raise ValueError("train_test_split needs an execution context")

    db = env.get_db()

    if cache_key is not None:
        data_split = db.data_split
        f = data_split.find_one({'key': cache_key, 'competition': competition})
        if f is not None and 'min_tool_coverage' in f:
            return None

    search = {
        'svcomp': competition
    }
    if category is not None:
        search['category'] = category

    filter_func = build_filter_func(filter)

    cur = db.svcomp.find(search, ['name', 'tool', 'category'])
    tool_count = {}
    abs_map = {}
    for sv_obj in tqdm(cur, total=cur.count()):
        name = sv_obj['name']
        if filter_func(name):
            tool = sv_obj['tool']
            if tool not in tool_count:
                tool_count[tool] = 0
            tool_count[tool] += 1
            if name not in abs_map:
                abs_map[name] = set([])
            abs_map[name].add(sv_obj['category'])

    abs_len = sum([len(v) for v in abs_map.values()])

    return {t: c / abs_len for t, c in tool_count.items()}


@task_definition()
def covered_tools(key, competition, coverage, min_coverage=0.8, env=None):
    if env is None:
        raise ValueError("train_test_split needs an execution context")

    db = env.get_db()
    data_split = db.data_split

    f = data_split.find_one({'key': key, 'competition': competition})

    if f is not None and 'min_tool_coverage' in f\
       and f['min_tool_coverage'] == min_coverage:
        return f['tools']

    tools = [t for t, c in coverage.items() if c >= min_coverage]
    data_split.update_one(
        {'key': key, 'competition': competition},
        {"$set": {'tools': tools, 'min_tool_coverage': min_coverage}}
    )
    return tools


@task_definition()
def filter_by_stat(competition, conditions, lesser=True, duplicate=True,
                   env=None):
    if env is None:
        raise ValueError("train_test_split needs an execution context")

    db = env.get_db()
    stat = db.graph_statistics

    op = '$lte' if lesser else '$gte'
    search = {
        'competition': competition
    }
    for key, cond in conditions.items():
        search[key] = {op: cond}

    if not duplicate:
        search['duplicate'] = {'$exists': 0}

    return [o['name'] for o in stat.find(search, ['name'])]


def get_svcomp_train_test(key, competition, category=None, test_ratio=0.1,
                          min_tool_coverage=0.8, ret_key=False):
    split = train_test(key, competition, category=category,
                       test_ratio=test_ratio)
    cov = tool_coverage(
        competition, filter=split[0], category=category,
        cache_key=key
    )
    cov_tools = covered_tools(
        key, competition, cov, min_coverage=min_tool_coverage
    )

    if ret_key:
        return cov_tools, split[0], split[1], key

    return cov_tools, split[0], split[1]


def get_svcomp_cv(key, competition, category=None, cv=10,
                  min_tool_coverage=0.8, filter=None, ret_key=False):
    split = train_crossval(key, competition, category=category,
                           cv=cv, filter=filter)
    split_it = tsk.fork(split)
    cov = tool_coverage(
        competition, filter=split_it[1], category=category,
        cache_key=split_it[0]
    )
    cov_tools = covered_tools(
        split_it[0], competition, cov, min_coverage=min_tool_coverage
    )

    if ret_key:
        return cov_tools, split_it[1], split_it[2], split_it[0]

    return cov_tools, split_it[1], split_it[2]


def index(x, y, n):
    """
    Return a single index for given coordinates.

    Map given coordinates to an index in a linear memoryself.
    It is important that (x, y) and (y, x) share the same index.
    The output ranges between 0 and n(n+1)/2.
    """
    if x >= n:
        raise ValueError('x: %d is out of range 0 to %d' % (x, n))
    if y >= n:
        raise ValueError('y: %d is out of range 0 to %d' % (y, n))
    if y == x:
        raise ValueError("tools aren't compared with themself")
    if x > y:
        tmp = y
        y = x
        x = tmp

    return int(x * (n - 0.5*(x+1))
               + (y - (x+1)))


def get_ranking(preference_vector, tool_order, allow_tie=True):
    count = {t: 0 for t in tool_order}
    n = len(tool_order)

    for i, t1 in enumerate(tool_order):
        for j in range(i + 1, len(tool_order)):
            t2 = tool_order[j]
            pref = preference_vector[
                index(i, j, n)
            ]
            count[t1] += pref
            count[t2] += 1 - pref

    buckets = {}

    for t, c in count.items():
        if c not in buckets:
            buckets[c] = t
        else:
            if not isinstance(buckets[c], list):
                buckets[c] = [buckets[c]]
            buckets[c].append(t)

    R = [t for c, t in sorted(buckets.items(),
                              key=lambda X: X[0],
                              reverse=True)]

    if not allow_tie:
        o = []
        for r in R:
            if isinstance(r, list):
                o.extend(r)
            else:
                o.append(r)
        R = o
    return R


def parse_label(status, time, ground_truth):
    if time >= 900:
        return "timeout", 900.0

    if ground_truth:
        if 'true' in status:
            return "success", time
    else:
        if 'false' in status:
            return "success", time

    if 'unknown' in status.lower():
        return "unknown", time

    return "fail", time


def get_labels(db, competition, category=None, filter=None):
    search = {
        'svcomp': competition
    }
    if category is not None:
        search['category'] = category

    filter_func = build_filter_func(filter)

    labels = {}
    cur = db.svcomp.find(search)
    for sv_obj in tqdm(cur, total=cur.count()):
        name = sv_obj['name']
        if filter_func(name):
            category = sv_obj['category']
            if category not in labels:
                labels[category] = {}
            cat_labels = labels[category]
            if name not in cat_labels:
                cat_labels[name] = {}

            tool = sv_obj['tool']
            ground_truth = sv_obj['ground_truth']
            status = sv_obj['status']
            time = sv_obj['cputime']

            if tool not in cat_labels[name]:
                cat_labels[name][tool] = {}
            D = cat_labels[name][tool]
            D['label'], D['time'] = parse_label(status, time, ground_truth)

    return labels


def get_single_label(db, name, competition, category=None):
    search = {
        'name': name,
        'svcomp': competition
    }
    if category is not None:
        search['category'] = category

    labels = {}
    for sv_obj in db.svcomp.find(search):
        tool = sv_obj['tool']
        ground_truth = sv_obj['ground_truth']
        status = sv_obj['status']
        time = sv_obj['cputime']
        cat = sv_obj['category']

        if cat not in labels:
            labels[cat] = {}
        cat_labels = labels[cat]

        if tool not in cat_labels:
            cat_labels[tool] = {}
        D = cat_labels[tool]
        D['label'], D['time'] = parse_label(status, time, ground_truth)

    return labels


def compare_labels(label1, time1, label2, time2):
    lookup = {
        'success': 1,
        'unknown': 2,
        'timeout': 3,
        'fail': 4
    }

    l1 = lookup[label1]
    l2 = lookup[label2]

    if l1 < l2:
        return 1.0
    if l2 < l1:
        return 0.0

    if time1 < time2:
        return 1.0
    if time2 < time1:
        return 0.0

    return 0.5


def get_preferences(labels, tools):

    n = len(tools)
    pref = np.zeros((int(n*(n-1)/2)))

    for i, t1 in enumerate(tools):
        for j in range(i + 1, len(tools)):
            t2 = tools[j]
            p = index(i, j, n)

            l1, time1 = 'fail', 0
            l2, time2 = 'fail', 0

            if t1 in labels:
                l1 = labels[t1]['label']
                time1 = labels[t1]['time']

            if t2 in labels:
                l2 = labels[t2]['label']
                time2 = labels[t2]['time']

            pref[p] = compare_labels(
                l1, time1, l2, time2
            )

    return pref
