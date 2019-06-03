from taskflow import SessionNotInitializedException
import taskflow.task as task
import taskflow.execution_handler as ex
from taskflow.symbolic import ForkResource

import networkx as nx
import copy
import uuid


def local_logger(text):
    print(text)


def _breaking_node(graph, id):
    if id == 'START' or id == 'STOP':
        return True

    if id.startswith('fork_') or id.startswith('merge_'):
        return False

    node = graph.node[id]
    attr = node['attributes']

    if 'workload' in attr:
        if attr['workload'] == 'heavy':
            return True

    if 'timeout' in attr:
        return True

    if 'optional' in attr:
        return True

    return False


def _break_seq(graph, src, target):

    if src.startswith('fork_'):
        return True

    if target.startswith('merge_'):
        return True

    if _breaking_node(graph, src):
        return True

    if _breaking_node(graph, target):
        return True

    if graph.out_degree(src) > 1:
        return True

    if graph.in_degree(target) > 1:
        return True

    return False


def _assign_to_seq(graph):
    assign = {}

    last_seq = 1

    seen = set(['START'])

    queue = [('START', 0)]

    while len(queue) > 0:
        node, seq = queue.pop()

        dests = []

        for _, v, dest in graph.out_edges(node, data="dest"):
            if v in seen:
                continue

            if dest is not None:
                dests.extend(dest)

            if _break_seq(graph, node, v):
                queue.append((v, last_seq))
                last_seq += 1
            else:
                queue.append((v, seq))

            seen.add(v)

        assign[node] = (seq, dests)

    ret_assign = {}

    for k, v in assign.items():
        nk = v[0]
        if nk not in ret_assign:
            ret_assign[nk] = []
        ret_assign[nk].append((k, v[1]))

    return assign, ret_assign


def _collapse_seq(graph):
    assign, group = _assign_to_seq(graph)

    seq_graph = nx.DiGraph()
    func_dict = {k: f for k, f in graph.nodes.items()}

    for i, V in group.items():
        last = len(V) - 1
        V[last] = (V[last][0], ['__out__'])
        seq_graph.add_node(i, sequence=V)

    for i, V in group.items():
        for _, n, dest in graph.out_edges([v[0] for v in V], data="dest"):
            seq = assign[n][0]
            if seq != i:
                seq_graph.add_edge(i, seq, dest=dest)

    seq_graph.graph['functions'] = func_dict
    seq_graph.graph['start'] = assign['START'][0]
    seq_graph.graph['stop'] = assign['STOP'][0]

    return seq_graph


def _seq_to_dot(graph):
    s = "digraph D {\n"

    for n in graph:
        seq = graph.nodes[n]['sequence']
        if len(seq) > 1:
            table = ""

            for el in seq:
                table = table + "* %s\\n" % el[0]

        else:
            table = seq[0][0]

        s = s + "N%i [label=\"%s\"];\n" % (n, table)

    for u, v in graph.edges():
        s = s + "N%i -> N%i;\n" % (u, v)

    s = s + "}"
    return s


def _backend_execute(function, kwargs, backend_setting):

    resource = None
    for k, variable in kwargs.items():
        if isinstance(variable, ForkResource):
            resource = variable

    if resource:
        keys = set([])
        for k, variable in kwargs.items():
            if isinstance(variable, ForkResource) and\
                    variable.src_ == resource.src_:
                keys.add(k)

        results = []
        for i in range(len(resource.obj_)):
            bind_kwargs = copy.copy(kwargs)
            for k in keys:
                bind_kwargs[k] = kwargs[k].obj_[i]
            results.append(
                _backend_execute(function, bind_kwargs, backend_setting)
            )
        return ForkResource(results, resource.src_)

    if '__logger__' not in backend_setting:
        backend_setting['__logger__'] = local_logger

    return ex.execute_function(function, kwargs, backend_setting)


def _handle_func_call(function, kwargs, backend_setting):

    if function == '__fork__':
        return ForkResource(kwargs['__fork__'])

    if function == '__merge__':
        res = []
        for k in kwargs:
            if k.startswith('__merge__'):
                res.append(ex._handle_merge(kwargs[k], backend_setting['flatten'])[0])
        if backend_setting['flatten']:
            out = []
            for r in res:
                out.extend(r)
            res = out
        return res

    args = {}

    for k, v in kwargs.items():
        if '::list_' in k:
            name, pos = k.rsplit('::list_', 1)
            if name not in args:
                args[name] = []
            arg_list = args[name]
            pos = int(pos)
            while len(arg_list) <= pos:
                arg_list.append([])
            arg_list[pos] = v
        elif '::dict_' in k:
            name, pos = k.rsplit('::dict_', 1)
            if name not in args:
                args[name] = {}
            args[name][pos] = v
        else:
            args[k] = v

    return _backend_execute(function, args, backend_setting)


def _handle_func_seq(sequence, initial_args):
    bind = initial_args

    for func in sequence:
        bind.update(func[1])
        result = _handle_func_call(func[0], bind, func[3])
        bind = {}
        for p_bind in func[2]:
            bind[p_bind] = result

    return bind


def _execute_node(seq_graph, id, binding):

    seq_node = seq_graph.nodes[id]

    execution_list = seq_node['sequence']

    if len(execution_list) == 0:
        return binding

    func_seq = []

    for func_id, bind_id in execution_list:
        if func_id == 'START':
            return {'__out__': None}

        if func_id == 'STOP':
            return {'__out__': [v for k, v in binding.items()
                                if k.startswith('end_')]}

        func_node = seq_graph.graph['functions'][func_id]

        if func_id.startswith('fork_'):
            func_seq.append(('__fork__', {}, bind_id, {}))
        elif func_id.startswith('merge_'):
            func_seq.append(('__merge__', {}, bind_id,
                            {'flatten': func_node['flatten']}))
        else:
            func_seq.append((
                func_node['function'], func_node['env'],
                bind_id, func_node['attributes']
            ))

    return _handle_func_seq(func_seq, binding)


def _execute_graph(seq_graph):

    queue = [0]

    bindings = {0: {}}

    while len(queue) > 0:
        id = queue.pop()
        bind = bindings[id]

        apply = True

        for _, _, dest in seq_graph.in_edges(id, data='dest'):
            if not apply:
                break
            if dest is None:
                break
            for D in dest:
                if D not in bind:
                    apply = False
                    break

        if not apply:
            continue

        result = _execute_node(seq_graph, id, bind)['__out__']

        for _, v, dest in seq_graph.out_edges(id, data='dest'):
            if v not in bindings:
                bindings[v] = {}
            if dest is not None:
                for D in dest:
                    bindings[v][D] = result
            queue.append(v)

    end_bind = bindings[seq_graph.graph['stop']]
    end_keys = sorted([k for k in end_bind.keys() if k.startswith('end_')])

    if len(end_keys) == 1:
        return end_bind[end_keys[0]]

    return [end_bind[k] for k in end_keys]


class LocalBackend:

    def __init__(self):
        self._sessions = {}
        self._session_result = {}

    def init(self, session_id):
        self._sessions[session_id] = None

    def attach(self, session_id, req_id, sequence_graph):
        if session_id not in self._sessions:
            raise SessionNotInitializedException()

        self._sessions[session_id] = sequence_graph

        if session_id in self._session_result:
            del self._session_result[session_id]

    def execute(self, session_id):

        if session_id in self._sessions:

            if session_id not in self._session_result:
                self._session_result[session_id] = _execute_graph(
                    self._sessions[session_id]
                )

            return True

        return False

    def retrieve_result(self, session, req_id):
        if session not in self._sessions:
            raise SessionNotInitializedException(
                "Session %s is not initialized!" % session
            )

        if session not in self._session_result:
            raise SessionNotInitializedException(
                "Something bad went wrong!"
            )

        return self._session_result[session]

    def cancel_session(self, session):
        if session in self._sessions:
            del self._sessions[session]
        if session in self._session_result:
            del self._session_result[session]


class Session(object):

    def __init__(self, session=None, auto_join=False, backend=LocalBackend()):
        self._backend = backend
        if session is not None:
            self._session_id = session
        self._auto_join = auto_join

    def __enter__(self):
        graph = nx.DiGraph()

        if not hasattr(self, '_session_id'):
            self._session_id = str(uuid.uuid4())

        self._backend.init(self._session_id)

        return SessionInstance(self._session_id, graph,
                               self._backend, self._auto_join)

    def __exit__(self, type, value, traceback):
        self._backend.cancel_session(
            self._session_id
        )


class SessionInstance(object):

    def __init__(self, session_id, graph, backend, auto_join=False):
        self._session_id = session_id
        self._graph = graph
        self._backend = backend
        self._auto_join = auto_join

    def _build_graph(self, symbolic):

        graph = self._graph

        graph.remove_edges_from(
            list(graph.in_edges('STOP'))
        )
        for sym in symbolic:
            graph = task.build_graph(sym, graph=graph)

        return graph

    def run(self, symbolic):

        if not isinstance(symbolic, list) and not isinstance(symbolic, tuple):
            symbolic = [symbolic]

        graph = self._build_graph(symbolic)

        req_id = str(uuid.uuid4())

        self._sequence_graph = _collapse_seq(graph)
        self._backend.attach(self._session_id, req_id, self._sequence_graph)

        if not self._backend.execute(self._session_id):
            raise ValueError(
                'Bad session submission!'
            )

        ret = SessionResult(self._session_id, req_id, self._backend)

        if self._auto_join:
            return ret.join()
        return ret


class SessionResult:

    def __init__(self, session_id, req_id, backend):
        self._session_id = session_id
        self._backend = backend
        self._req_id = req_id

    def join(self, callback=None):
        result = self._backend.retrieve_result(
            self._session_id, self._req_id
        )

        if isinstance(result, list) and len(result) == 1:
            result = result[0]

        if callback is not None:
            callback(result)
        else:
            return result

    def __repr__(self):
        return "SessionResult( id = %s )" % self._session_id


def openLocalSession(session_id=None, auto_join=False):
    return Session(session=session_id, auto_join=auto_join)
