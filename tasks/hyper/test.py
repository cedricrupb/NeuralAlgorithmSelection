from tasks.utils import train_utils
from tasks.data import proto_data
from tasks.torch.execute import execute_model
from taskflow import backend


def overfit_model(config):
    dataset = config['dataset']
    ast_type = "bag"

    if 'ast_type' in dataset:
        ast_type = dataset['ast_type']
        del dataset['ast_type']

    ast_type = ast_type == 'bag'

    tools, train, test = train_utils.get_svcomp_train_test(
        **dataset
    )
    dataset = proto_data.download_lmdb(
        tools, dataset['competition'],
        train[0:32], test[0:1], category=dataset['category'],
        ast_bag=ast_type
    )
    return execute_model(tools, config, dataset)


if __name__ == '__main__':
    config = {
        'model': {
            'node_input': 148,
            'edge_input': 3,
            'global_input': 4,
            'modules': {
                'm0': {'type': 'tasks::Embedding', 'node_dim': 32},
                'm1': {'type': 'torch::Linear', 'node_dim': 91},
                'm2': 'tasks::cga'
            },
            'bind': [
                ['source', 'x', 'x', 'm0'],
                ['m0', 'forward', 'x', 'm2'],
                ['source', 'batch', 'batch', 'm2'],
                ['source', 'category', 'condition', 'm2'],
                ['m2', 'forward', 'input', 'm1'],
                ['m1', 'forward', 'input', 'sink']
            ]
        },
        'dataset': {
            'key': '2019_all_categories_all_10000',
            'competition': '2019',
            'category': None,
            'test_ratio': 0.2,
            'min_tool_coverage': 0.8,
            'ast_type': 'bag'
        },
        'train': {
            'loss': 'masked::HingeLoss',
            'epoch': 2000,
            'batch': 32,
            'shuffle': True,
            'augment': False,
            'optimizer': {'type': 'torch::Adam', 'lr': 0.01,
                          'betas': [0.9, 0.98],
                          'eps': 1e-9},
            'scheduler': {
                'type': 'torch::CosineAnnealingLR', 'mode': 'epoch',
                'T_max': 50, 'eta_min': 0.001
            }
        },
        'test': {'type': 'category', 'scores': 'spearmann'}
    }
    train_test = overfit_model(config)
    with backend.openLocalSession() as sess:
        sess.run(train_test)
