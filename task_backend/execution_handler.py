from backend import ForkResource, _handle_merge
import copy


def execute_function(function, kwargs, backend_setting):
    if function == '__fork__':
        return ForkResource(kwargs['__fork__'])

    if function == '__merge__':
        res = []
        for k in kwargs:
            if k.startswith('__merge__'):
                res.append(_handle_merge(kwargs[k], backend_setting['flatten'])[0])
        if backend_setting['flatten']:
            out = []
            for r in res:
                out.extend(r)
            res = out
        return res

    return _single_execution(function, kwargs, backend_setting)


def _single_execution(function, kwargs, backend_setting):
    print((function, kwargs))
    # TODO add timeout etc.
    return function(**kwargs)
