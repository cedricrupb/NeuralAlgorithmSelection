import taskflow.symbolic as sym


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
