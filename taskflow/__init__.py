import taskflow.symbolic as sym
from inspect import signature


class EnvironmentException(Exception):
    pass


class SessionNotInitializedException(Exception):
    pass


class RemoteError(Exception):
    pass


class TimeoutException(Exception):
    pass


def fork(task):
    return sym.SymbolicForkElement(task)


def merge(join_list, flatten=False):
    return sym.SymbolicMergeElement(join_list, flatten)


def optional(func):
    return sym.optional(func)


def _expand_var(key, value):
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

                for k, x in _expand_var(k, x):
                    if isinstance(x, sym.Symbolic):
                        task_args_[k] = x
                    else:
                        env_args_[k] = x

            return sym.SymbolicFunction(self._function, version,
                                        self._signature,
                                        env_args_, task_args_,
                                        def_args)

    return task_definition_inner
