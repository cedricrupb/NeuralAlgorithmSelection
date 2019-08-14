import taskflow as tsk
from taskflow import task_definition, backend

import torch as th

from tasks.utils import train_utils
from tasks.data import proto_data

from tasks.torch.visdom import VisdomLogger
from tasks.torch.train import build_training, print_log
from tasks.torch.graph import build_graph
from tasks.torch.test import build_test
from tasks.torch.model_config import partial_to_model, micro_to_partial
from gridfs import GridFS

import time
import os
import shutil
import numpy as np


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
        if m['score'] < state['score']:
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


def build_visdom_log(config):

    def void_log(epoch, real_it, loss, val_loss, val_score):
        pass

    if 'name' not in config:
        return void_log

    name = config['name']
    visdom = VisdomLogger(env=config['dataset']['key'])

    print("Start Visdom with name %s." % name)

    def viz_log(epoch, real_it, loss, val_loss, val_score):

        if val_score is not None:
            score = float(val_score)
            visdom.log_acc(name, epoch, score)

        if val_loss is not None:
            val_loss = float(val_loss)

        visdom.log_loss(name, real_it, float(loss), val_loss)

    return viz_log


@task_definition()
def execute_model(tools, config, dataset_path, name=None, ret=None, env=None):

    if name is not None:
        config['name'] = name

    model_config = config['model']

    if 'type' in model_config:
        model_config = micro_to_partial(model_config)

    if 'layers' in model_config:
        model_config = partial_to_model(model_config, dataset_path)

    model = build_graph(model_config).compile()
    train = build_training(config['train'], model)
    test = build_test(config['test'])
    config['model'] = model_config

    visdom = build_visdom_log(config)

    print(train)

    best = 0
    save, load = build_model_io(env.get_cache_dir())

    start_time = time.time()
    max_it = 0
    for epoch, it, train_loss, val_loss, val_score in train.train_iter(
                                                        tools, dataset_path
                                                    ):

        max_it = max(max_it, it)
        real_it = max_it*epoch + it
        visdom(epoch, real_it, train_loss, val_loss, val_score)

        if val_loss is not None:
            print_log(
                epoch, it, train_loss, val_loss, val_score
            )
            print("Time: %f sec" % (time.time() - start_time))
            start_time = time.time()

            if val_score > best:
                best = val_score
                # save({
                #    'model': train.optimizer.model,
                #    'epoch': epoch,
                #    'score': val_score
                # })

    # model = load({'model': train.optimizer.model,
    #              'score': 0})['model']

    test_res = test(tools, model, dataset_path)

    test_res['num_params'] = sum(p.numel() for p in model.parameters())

    if 'name' in config:
        store_model(
            env.get_db(), config, model, tools,
            env.get_cache_dir(), test_res
        )

    if ret is not None and ret in test_res:
        return test_res[ret]['mean'], test_res[ret]['std']


def build_filter(competition, filter_config):
    condition = {}
    for key, limit in filter_config.items():
        condition[key] = limit

    filter = train_utils.filter_by_stat(competition, condition)
    return filter


def build_model(config, ret=None):
    dataset = config['dataset']
    ast_type = "bag"

    if 'ast_type' in dataset:
        ast_type = dataset['ast_type']
        del dataset['ast_type']

    ast_type = ast_type == 'bag'

    filter = None

    if 'filter' in dataset:
        filter = build_filter(dataset['competition'], dataset['filter'])
        del dataset['filter']

    tools, train, test = train_utils.get_svcomp_train_test(
        **dataset
    )

    dataset = proto_data.download_lmdb(
        tools, dataset['competition'],
        train, test, category=dataset['category'],
        ast_bag=ast_type, filter=filter
    )
    return execute_model(tools, config, dataset, ret=ret)


if __name__ == '__main__':
    res = []

    for i in range(1):

        config = {
            'name': 'dense12_200_memory_%i' % i,
            'model': {
                "type": "dense_gin",
                "embed_size": 32,
                "growth": 32,
                "layers": 2,
                "out": 96,
                "global_condition": True,
                "global_norm": True
            },
            'dataset': {
                'key': 'rank18_memory_%i' % i,
                'competition': '2018',
                'category': 'memory',
                'test_ratio': 0.2,
                'min_tool_coverage': 0.8,
                'ast_type': 'bag'
            },
            'train': {
                'loss': 'masked::HingeLoss',
                'epoch': 40,
                'batch': 32,
                'shuffle': 42,
                'augment': False,
                'clip_grad': 5,
                'optimizer': {'type': 'torch::AdamW', 'lr': 0.01,
                              'weight_decay': 1e-4},
                'scheduler': {
                    'type': 'torch::StepLR', 'mode': 'epoch',
                    'step_size': 20, 'gamma': 0.5
                },
                'validate': {
                    'checkpoint_step': 0,
                    'score': 'spearmann',
                    'split': 0.1
                }
            },
            'test': {'type': 'category', 'scores': 'spearmann'}
        }

        train_test = build_model(config, ret='spearmann')
        with backend.openLocalSession() as sess:
            mean, _ = sess.run(train_test).join()
        print("Spearmann: %f" % mean)
        res.append(mean)

print("Acc: %f (+- %f)" % (np.mean(res), np.std(res)))
