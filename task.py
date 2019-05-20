import networkx as nx
from inspect import signature
import uuid
import hashlib


def expand_var(key, value):
    if isinstance(value, Symbolic):
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
                if isinstance(x, optional):
                    raise ValueError("Optional binding are only allowed for optional arguments!")

            bound_args = self._signature.bind(*args, **kwargs)

            env_args_ = {}
            task_args_ = {}

            for k in bound_args.arguments:
                x = bound_args.arguments[k]

                for k, x in expand_var(k, x):
                    if isinstance(x, Symbolic):
                        task_args_[k] = x
                    else:
                        env_args_[k] = x

            return SymbolicFunction(self._function, version,
                                    self._signature,
                                    env_args_, task_args_,
                                    def_args)

    return task_definition_inner


@task_definition()
def get_item(obj, key):
    return obj[key]


def hash_str(string):
    bytes = string.encode('utf-8')
    h = hashlib.blake2b(digest_size=32)
    h.update(bytes)
    return str(h.hexdigest())


class Symbolic(object):

    def __getitem__(self, key):
        return get_item(self, key)


class optional(Symbolic):

    def __init__(self, obj):
        self._obj = obj

    def get_content(self):
        return self._obj

    def __local__(self, *args, **kwargs):
        return self._obj.__local__(*args, **kwargs)

    def __str__(self):
        return "optional(%s)" % str(self._obj)


class SymbolicFunction(Symbolic):

    def __init__(self, func, version, signature, env_args,
                 task_args, attr=None):
        self.function_ = func
        self.version_ = version
        self._signature = signature
        self.env_args_ = env_args
        self.task_args_ = task_args
        self.attr_ = attr

    def __exec__(self, **kwargs):
        return self.function_(**kwargs)

    def __identifier__(self):
        param = []

        for k in self._signature.parameters:
            if k in self.env_args_:
                param.append(str(self.env_args_[k]))
            elif k in self.task_args_:
                param.append("symbolic_"+hash_str(str(self.task_args_[k])))

        param = ', '.join(param)

        return self.function_.__name__+"("+param+")"

    def __hash__(self):

        param = []

        for k in self._signature.parameters:
            if k in self.env_args_:
                param.append(str(self.env_args_[k]))
            elif k in self.task_args_:
                param.append(hash_str(str(self.task_args_[k])))

        param = ', '.join(param)

        return hash_str(self.function_.__name__+"_"+param+"::"+self.version_)

    def __str__(self):
        kwargs = {}
        kwargs.update(self.env_args_)
        kwargs.update(self.task_args_)

        param = []

        for k in self._signature.parameters:
            if k in kwargs:
                param.append(str(kwargs[k]))

        param = ', '.join(param)

        return "symbolic ver. %s [ %s (%s) ]" % (self.version_,
                                                 self.function_.__name__,
                                                 param)


class SymbolicForkElement(Symbolic):

    def __init__(self, list):
        self.list_ = list

    def __identifier__(self):
        return "fork_"+self.list_.__identifier__()

    def __str__(self):
        return "element of %s" % str(self.list_)


class SymbolicMergeElement(Symbolic):

    def __init__(self, args, flatten=False):
        self.args_ = {
            '__merge__%i' % i: v for i, v in enumerate(args)
        }
        self.flatten_ = flatten

    def __identifier__(self):
        return "merge_"+str([v.__identifier__() for v in self.args_.values()])

    def __str__(self):
        return "merge of %s" % str([str(v) for v in self.args_.values()])


def fork(task):
    return SymbolicForkElement(task)


def merge(join_list, flatten=False):
    return SymbolicMergeElement(join_list, flatten)


def build_graph(symbolic_func, call=None, callDest=None, graph=None):

    if graph is None:
        graph = nx.DiGraph()
        graph.add_node("START")

    if call is None:
        call = "STOP"
        graph.add_node(call)
        callDest = 'end_%s' % str(uuid.uuid1())

    if not isinstance(symbolic_func, Symbolic):
        return graph

    id = symbolic_func.__identifier__()

    if isinstance(symbolic_func, SymbolicForkElement):
        args = {}
        dependencies = {"__fork__": symbolic_func.list_}
    elif isinstance(symbolic_func, SymbolicMergeElement):
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
