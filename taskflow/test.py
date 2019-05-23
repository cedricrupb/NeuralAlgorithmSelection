import taskflow as tsk
from taskflow import task_definition
from taskflow.task import build_graph, to_dot

from taskflow.backend import openLocalSession

import copy


@task_definition()
def dummy_supplier():
    return ['task_1.c', 'task_2.c', 'task_3.c', 'task_4.c']


@task_definition()
def process2graph(path):
    return 'graph (%s)' % path


@task_definition()
def collect_subgraphs(graph, labels={}, opt=None):
    labels = copy.deepcopy(labels)
    labels[graph] = "Ahoi"
    return 'subgraph_%s' % graph, labels


@task_definition()
def unpack_first(L):
    return [l for i, l in enumerate(L) if i % 2 == 0]


@task_definition()
def task_sort(L):
    return sorted(L)


@task_definition()
def task_print_list(x):
    for e in x:
        print(e)
    return x


@task_definition()
def range_task(number):
    return [{'number': i} for i in range(number)]


@task_definition()
def error():
    raise ValueError("Test error")


def dummy_composite():
    sup = dummy_supplier()
    sup_fork = tsk.fork(sup)
    collect = collect_subgraphs(sup_fork)
    y_it = tsk.fork(range_task(10))
    collect_1 = collect_subgraphs(collect[0], collect[1], opt=tsk.optional(error()))
    join = tsk.merge([collect_1], flatten=True)
    return task_print_list(
        join
    )


if __name__ == '__main__':
    comp = dummy_composite()

    with openLocalSession(auto_join=True) as sess:
        sess.run(comp)
