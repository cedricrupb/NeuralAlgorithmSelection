import networkx as nx
from inspect import signature
import uuid
import hashlib

import taskflow.symbolic as sym


def expand_var(key, value):
    if isinstance(value, sym.Symbolic):
        yield key, value
        return

    if isinstance(value, str):
        yield key, value
        return

    try:
        for k, v in value.items():
            yield key+"::dict_"+str(k), v
    except (AttributeError, TypeError):
        try:
            for i, v in enumerate(value):
                yield key+"::list_%i" % i, v
        except (AttributeError, TypeError):
            yield key, value


def task_definition(version="1.0", **def_args):
    class task_definition_inner(object):

        def __init__(self, func, return_type=None):
            self._function = func
            self._signature = signature(func)

        def __call__(self, *args, **kwargs):

            for x in args:
                if isinstance(x, sym.optional):
                    raise ValueError("Optional binding are only allowed for optional arguments!")

            bound_args = self._signature.bind(*args, **kwargs)

            env_args_ = {}
            task_args_ = {}

            for k in bound_args.arguments:
                x = bound_args.arguments[k]

                for k, x in expand_var(k, x):
                    if isinstance(x, sym.Symbolic):
                        task_args_[k] = x
                    else:
                        env_args_[k] = x

            return sym.SymbolicFunction(self._function, version,
                                        self._signature,
                                        env_args_, task_args_,
                                        def_args)

    return task_definition_inner


@task_definition()
def get_item(obj, key):
    return obj[key]


def build_graph(symbolic_func, call=None, callDest=None, graph=None):

    if graph is None:
        graph = nx.DiGraph()
        graph.add_node("START")

    if call is None:
        call = "STOP"
        graph.add_node(call)
        callDest = 'end_%s' % str(uuid.uuid1())

    if not isinstance(symbolic_func, sym.Symbolic):
        return graph

    id = symbolic_func.__identifier__()

    if isinstance(symbolic_func, sym.SymbolicForkElement):
        args = {}
        dependencies = {"__fork__": symbolic_func.list_}
    elif isinstance(symbolic_func, sym.SymbolicMergeElement):
        args = {'flatten': symbolic_func.flatten_}
        dependencies = symbolic_func.args_
    else:
        args = {
            'env': symbolic_func.env_args_,
            'function': symbolic_func.function_,
            'dependency_vars': list(symbolic_func.task_args_.keys()),
            'version': symbolic_func.version_,
            'attributes': symbolic_func.attr_
        }
        dependencies = symbolic_func.task_args_

    if not graph.has_node(id):
        graph.add_node(id, **args)

        if len(dependencies) == 0:
            graph.add_edge("START", id)
        else:
            for k, symbolic_dep in dependencies.items():
                build_graph(symbolic_dep, call=id,
                            callDest=k, graph=graph)

    if not graph.has_edge(id, call):
        graph.add_edge(id, call, dest=[])
    graph.edges[id, call]['dest'].append(callDest)

    return graph


def to_dot(graph):
    print("digraph D {")
    index = {}
    counter = 0

    for n in graph:
        index[n] = "N"+str(counter)
        counter += 1

        print(index[n]+"[label=\"%s\"];" % n)

    for n1, n2 in graph.edges():
        print(index[n1]+" -> "+index[n2]+";")

    print("}")
