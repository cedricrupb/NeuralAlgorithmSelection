import taskflow as tsk
from taskflow import task_definition, backend
from tasks.hyper.exec_handler import cfg_egin_execute

import copy


grid = {
  "model_depth": [0, 1, 2, 3, 4, 5],
  "embed_size": [32, 64],
  "hidden_size": [16, 32, 64],
  "batch_size": [32, 128],
  "loss_func": ["hinge", "bce"]
}


def num_to_pos(num, max_c, order):
    pos = []
    c = num
    for i, o in enumerate(order):
        m = max_c[o]
        pos.append(c % m)
        c = int(c / m)
    return pos


def grid_supply():

    max_c = {k: len(v) for k, v in grid.items()}
    num = 1

    for v in max_c.values():
        num *= v

    order = [k for k in grid.keys()]

    out = []

    for i in range(num):
        pos = num_to_pos(i, max_c, order)

        out.append({
            o: grid[o][pos[j]]
            for j, o in enumerate(order)
        })

    return out


@task_definition()
def grid_supplier(base_name, dataset_key, competition, epoch=50,
                  category=None):

    base_cfg = {
        'dataset_key': dataset_key,
        'competition': competition,
        'category': category,
        'epoch': epoch
    }

    out = []
    for i, cfg in enumerate(grid_supply()):
        cfg.update(base_cfg)
        cfg['name'] = '%s_%i' % (base_name, i)
        out.append(cfg)

    return out


if __name__ == '__main__':
    grid_sup = grid_supplier(
                'all_10000_test', '2019_all_categories_all_10000',
                '2019')
    grid_fork = tsk.fork(grid_sup)
    egin = cfg_egin_execute(grid_fork)
    with backend.openLocalSession(auto_join=True) as sess:
        sess.run(egin)
