import taskflow as tsk
from taskflow import task_definition, backend

import torch as th

from tasks.utils import train_utils
from tasks.data import proto_data

from tasks.torch.train import build_training, print_log
from tasks.torch.graph import build_graph
from tasks.torch.test import build_test
from tasks.torch.model_config import partial_to_model, micro_to_partial
from gridfs import GridFS

import time
import os
import shutil


def build_model_io(base_dir):

    def identity(state, filename=""):
        return state

    if base_dir is None:
        return identity, identity

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    def save(state, filename="checkpoint.pth.tar"):
        path = os.path.join(base_dir, filename)
        th.save(state, path)

    def load_best(state, filename="checkpoint.pth.tar"):
        path = os.path.join(base_dir, filename)
        m = th.load(path)
        if m['score'] > state['score']:
            return m
        else:
            return state

    return save, load_best


def store_model(db, config, model, tools, base_path, eval):
    base_path = os.path.join(base_path, "model.th")
    th.save(model.state_dict(), base_path)

    model_key = config['name']
    dataset = config['dataset']['key']

    fs = GridFS(db)
    file = fs.new_file(model=model_key,
                       dataset_key=dataset,
                       app_type='torch_model',
                       encoding="utf-8")

    try:
        with open(base_path, "rb") as i:
            shutil.copyfileobj(i, file)
    finally:
        file.close()

    insert = {
        'experiment': model_key,
        'dataset': config['dataset']['key'],
        'competition': config['dataset']['competition'],
        'category': config['dataset']['category'],
        'experiment_def': config,
        'model_ref': file._id,
        'tools': tools
    }

    insert.update(eval)

    models = db.torch_experiment
    models.insert_one(insert)


@task_definition()
def execute_model(tools, config, dataset_path, ret=None, env=None):

    model_config = config['model']

    if 'type' in model_config:
        model_config = micro_to_partial(model_config)

    if 'layers' in model_config:
        model_config = partial_to_model(model_config, dataset_path)

    model = build_graph(model_config).compile()
    train = build_training(config['train'], model)
    test = build_test(config['test'])
    config['model'] = model_config

    print(train)

    best = 0.0
    save, load = build_model_io(env.get_cache_dir())

    start_time = time.time()
    for epoch, it, train_loss, val_loss, val_score in train.train_iter(
                                                        tools, dataset_path
                                                    ):
        if val_loss is not None:
            print_log(
                epoch, it, train_loss, val_loss, val_score
            )
            print("Time: %f sec" % (time.time() - start_time))
            start_time = time.time()

            if val_score > best:
                best = val_score
                save({
                    'model': train.optimizer.model,
                    'epoch': epoch,
                    'score': val_score
                })

    model = load({'model': train.optimizer.model,
                  'score': 0.0})['model']

    test_res = test(tools, model, dataset_path)

    if 'name' in config:
        store_model(
            env.get_db(), config, model, tools,
            env.get_cache_dir(), test_res
        )

    if ret is not None and ret in test_res:
        return test_res[ret]['mean']


def build_model(config):
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
        train, test, category=dataset['category'],
        ast_bag=ast_type
    )
    return execute_model(tools, config, dataset)


if __name__ == '__main__':

    config = {
        'name': 'test',
        'model': {
            'layers': [
                {'type': 'embed', 'node_dim': 32},
                {'type': 'edge_gin',
                 'node_dim': 32,
                 'hidden': 32,
                 'dropout': 0.1,
                 'norm': True
                 }
            ],
            'readout': [
                {'type': 'cga'},
                {'type': 'cga'}
            ]
        },
        'dataset': {
            'key': '2019_all_categories_all_10000',
            'competition': '2019',
            'category': 'reachability',
            'test_ratio': 0.2,
            'min_tool_coverage': 0.8,
            'ast_type': 'bag'
        },
        'train': {
            'loss': 'tasks::Rank_BCE',
            'epoch': 200,
            'batch': 32,
            'shuffle': True,
            'augment': False,
            'optimizer': {'type': 'torch::Adam', 'lr': 0.01,
                          'betas': [0.9, 0.98],
                          'eps': 1e-9},
            'scheduler': {
                'type': 'torch::StepLR', 'mode': 'epoch',
                'step_size': 50, 'gamma': 0.5
            },
            'validate': {
                'checkpoint_step': 0,
                'score': 'spearmann',
                'split': 0.1
            }
        },
        'test': 'spearmann'
    }

    train_test = build_model(config)
    with backend.openLocalSession() as sess:
        sess.run(train_test)
