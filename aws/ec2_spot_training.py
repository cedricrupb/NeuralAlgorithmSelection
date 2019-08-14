import argparse
import json
import execute as exc
import os


def load_model_list():
    if not os.path.exists("model_list.txt"):
        print("Require a model_list to run")

    with open("model_list.txt", "r") as i:
        return [r[:-1] for r in i.readlines()]


def load_task_list():
    if not os.path.exists("task_list.txt"):
        print("Require a task_list to run")

    with open("task_list.txt", "r") as i:
        tasks = [r[:-1] for r in i.readlines()]
        train = []
        test = []
        for t in tasks:
            tr, te = t.split('; ')
            train.append(tr)
            test.append(te)
        return train, test


def get_path(checkpoint, task, model):
    return os.path.join(checkpoint, task, model)


def find_todos(checkpoint):
    models = load_model_list()
    train, test = load_task_list()

    for i, t in enumerate(train):
        for m in models:
            path = get_path(checkpoint, t, m)
            if not os.path.isdir(path):
                os.makedirs(path)
            if not os.path.exists(os.path.join(path, "test.json")):
                yield (t, test[i], m)


def run_bench(checkpoint, dataset):

    path_pattern = os.path.join(dataset, "%s.zip")
    model_pattern = "%s.json"

    for train, test, model in find_todos(checkpoint):
        print("Start job. Train: %s, Test: %s, Model: %s" % (train, test, model))

        train_path = path_pattern % train

        if not os.path.isfile(train_path):
            print("Cannot start training. %s doesn't exist." % train_path)
            continue

        test_path = path_pattern % test
        if not os.path.isfile(test_path):
            print("Cannot start testing. %s doesn't exist." % test_path)
            continue

        model_path = model_pattern % model
        if not os.path.isfile(model_path):
            print("Cannot load model. %s doesn't exist." % model_path)
            continue

        exc.run(
            model_path, train_path, test_path,
            checkpoint=os.path.join(checkpoint, train)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("checkpoint_path")

    args = parser.parse_args()

    run_bench(
        args.checkpoint_path,
        args.dataset_path
    )
