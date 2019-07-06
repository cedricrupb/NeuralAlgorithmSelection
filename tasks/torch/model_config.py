import copy
from tasks.data import proto_data


def layer_module(config):
    config = copy.deepcopy(config)
    type = config['type']
    del config['type']

    if type == 'embed':
        config['type'] = 'tasks::Embedding'
        return config, ['x']

    if type == 'edge_gin':
        cfg = {'type': 'tasks::CEdgeGIN', 'node_dim': config['node_dim']}
        gin = {}
        if 'hidden' in config:
            gin['hidden'] = config['hidden']
            if 'dropout' in config:
                gin['dropout'] = config['dropout']
            if 'norm' in config:
                gin['norm'] = config['norm']
            else:
                gin['norm'] = False
        cfg['gin_nn'] = gin
        edge = {}
        if 'edge_hidden' in config:
            edge['hidden'] = config['edge_hidden']
            if 'edge_dropout' in config:
                edge['dropout'] = config['edge_dropout']
            if 'edge_norm' in config:
                edge['norm'] = config['edge_norm']
            else:
                edge['norm'] = False
        cfg['edge_nn'] = edge
        return cfg, ['x', 'edge_index', 'edge_attr']

    raise ValueError("Unknown type: %s" % type)


def readout_module(type):

    if type == 'cga':
        return 'tasks::cga'

    if type == 'add':
        return 'geo::global_sum_pool'

    if type == 'max':
        return 'geo::global_max_pool'

    return type


def layered_to_model(config):

    idc = 0
    layers = []
    readouts = []
    modules = {}
    bind = []
    current = 'source'

    for i, L in enumerate(config['layers']):
        id = 'm%i' % idc
        idc += 1
        layers.append(id)
        modules[id], req = layer_module(L)

        for r in req:
            if r == 'x':
                out = 'forward'
                if current == 'source':
                    out = 'x'
                bind.append([
                    current, out, 'x', id
                ])
            else:
                bind.append([
                    'source', r, r, id
                ])
        current = id

    for i, L in enumerate(config['readout']):
        type = L['type']
        of = i
        if 'of' in L:
            of = L['of']
        cond = []
        if 'cond' in L:
            cond = L['cond']

        id = 'm%i' % idc
        idc += 1
        modules[id] = readout_module(type)

        bind.append(['source', 'batch', 'batch', id])

        if type == 'cga':
            bind.append(['source', 'category', 'condition', id])

        bind.append([layers[of], 'forward', 'x', id])

        for pos in cond:
            if pos < len(readouts) and type == 'cga':
                bind.append([
                    readouts[pos], 'forward', 'condition', id
                ])
        readouts.append(id)

    for r in readouts:
        bind.append(
            [r, 'forward', 'input', 'sink']
        )

    return {'modules': modules, 'bind': bind}


def get_info(dataset_path):

    dataset = proto_data.GraphDataset(
        dataset_path, 'train', shuffle=False
    )
    example = dataset[0]

    info = {
        'node_input': example.x.shape[1],
        'edge_input': example.edge_attr.shape[1],
        'global_input': example.category.shape[1],
        'y': example.y.shape[1]
    }
    return info


def partial_to_model(config, dataset_path):

    if 'layers' in config:
        config = layered_to_model(config)

    info = get_info(dataset_path)
    out = info['y']
    del info['y']

    config.update(info)

    drop_id = 'out_drop'
    lin_id = 'out_lin'

    config['modules'][drop_id] = {
        'type': 'torch::Dropout', 'p': 0.1
    }
    config['modules'][lin_id] = {
        'type': 'torch::Linear', 'node_dim': out
    }
    config['bind'].append([drop_id, 'forward', 'input', lin_id])

    bind = []

    for B in config['bind']:
        if B[3] == 'sink':
            bind.append([B[0], B[1], 'input', drop_id])
        else:
            bind.append(B)
    bind.append([lin_id, 'forward', 'input', 'sink'])
    config['bind'] = bind

    return config


def micro_to_partial(config):
    type = config['type']

    if type == 'edge_cga':
        layers = config['layer']
        num_layer = int((len(layers)+1)/2)
        Ls = []
        readout = []
        for i in range(num_layer):
            if i == 0:
                Ls.append({
                    'type': 'embed',
                    'node_dim': layers[0]
                })
            else:
                p = i * 2 - 1
                hid = layers[p]
                out = layers[p + 1]
                Ls.append({
                    'type': 'edge_gin',
                    'node_dim': out,
                    'hidden': hid,
                    'dropout': 0.1,
                    'norm': True
                })
            readout.append({'type': 'cga'})
        return {
            'layers': Ls,
            'readout': readout
        }

    raise ValueError("Unknown Type: %s" % type)


if __name__ == '__main__':

    config = {
        'type': 'edge_cga',
        'layer': [32, 16, 8]
    }

    print(layered_to_model(micro_to_partial(config)))
