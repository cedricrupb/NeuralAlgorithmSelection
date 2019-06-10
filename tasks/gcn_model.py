import datetime
import os
import numpy as np
import graph_nets as gn
import sonnet as snt
from graph_nets import utils_tf
from graph_nets import utils_np
import tensorflow as tf


def load_model(model_key, model_args):

    def model(layer):
        reducer = tf.unsorted_segment_sum
        toglobal = gn.blocks.NodesToGlobalsAggregator(reducer)
        return toglobal(layer)

    return model


def placeholder_and_feed(generator):

    iterator = iter(generator)
    base_iter = next(iterator)
    first_g, first_l = base_iter

    place_graphs = utils_tf.placeholders_from_data_dicts(first_g)
    place_labels = tf.placeholder(tf.float32, name="label",
                                  shape=(None, first_l.shape[1]))

    def feed_gen():
        it = base_iter
        while True:
            if it == 'STOP':
                yield it

            graphs, labels = it
            graphs = utils_np.data_dicts_to_graphs_tuple(
                graphs
            )
            graphs = utils_tf.make_runnable_in_session(
                graphs
            )
            graphs = utils_tf.get_feed_dict(
                place_graphs, graphs
            )
            graphs[place_labels] = labels.astype(np.float32)
            yield graphs
            it = next(iterator)
    return place_graphs, place_labels, feed_gen


def graph_constant(dataset, labels):

    y = tf.constant(labels)

    graphs = utils_tf.data_dicts_to_graphs_tuple(
        dataset
    )
    graphs = utils_tf.make_runnable_in_session(
        graphs
    )

    return graphs, y


def stack_rank_model(base_model, output_size):

    initializers = {"w": tf.truncated_normal_initializer(stddev=1),
                    "b": tf.truncated_normal_initializer(stddev=1)}
    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.0001),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.01)}

    with tf.name_scope("rank_model"):
        linear_mod = snt.Linear(output_size=output_size,
                                initializers=initializers,
                                regularizers=regularizers)
        logistic_reg = snt.Sequential([linear_mod, tf.sigmoid])
        lr_sig = logistic_reg(base_model)

    return lr_sig


def loss_function(logits_true, logits_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=logits_true, logits=logits_pred
    )


def train_model(model_key, train_generator, validate_set,
                model_args=None, env=None):

    X, y, feed_gen = placeholder_and_feed(train_generator)
    # X_val, y_val = graph_constant(*validate_set)

    base_model = load_model(model_key, model_args)(X)
    logit_pred = stack_rank_model(base_model, y.shape[1])

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_regularization_loss = tf.reduce_sum(graph_regularizers)

    with tf.name_scope("cost_function") as scope:
        epsilon = 0.0001
        logits_ = tf.clip_by_value(logit_pred, epsilon, 1-epsilon)
        logits_ = tf.log(logits_ / (1 - logits_))
        cross_entropy = loss_function(
            y, logit_pred
        )

        cost = tf.reduce_mean(cross_entropy) + total_regularization_loss
        tf.summary.scalar("mean_entropy", cost)

    optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)
    merge_op = tf.summary.merge_all()

    time_string = datetime.datetime.now().isoformat()

    name = f"MLP_Test_{time_string}"
    avg_loss = []

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        summary_writer = tf.summary.FileWriter(os.path.join(env.get_cache_dir(), name), graph=sess.graph)

        print("Epoch 0:")
        ep = 0
        i = 0
        for feed_dict in feed_gen():

            if feed_dict == 'STOP':
                ep += 1
                i = 0
                print("Epoch %i:" % ep)

            _, loss = sess.run([optimizer, cost], feed_dict=feed_dict)
            print("Iteration %i Loss %f" % (i, loss))
            avg_loss.append(loss)
            if i % 10 == 0:
                loss = np.mean(avg_loss)
                print("Iteration %i Loss %f" % (i, loss))
                avg_loss = []

            summary_str = sess.run(merge_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            i += 1



def prepare_placeholder(train):

    first = train.take(1)
    iter = first.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        initial = sess.run([iter])[0]

    label_length = initial['preference'].shape[0]

    dummy = {
        'globals': [0., 0., 1., 0.],
        'nodes': [[0.2]*148],
        'edges': [[1., 0., 0., 0.]],
        'senders': [0],
        'receivers': [0]
    }

    place_graphs = utils_tf.placeholders_from_data_dicts([dummy])
    place_labels = tf.placeholder(tf.float32, name="label",
                                  shape=(None, label_length))

    return place_graphs, place_labels


def build_dense(index_0, index_1, values):
    pass


def batch_load(sess, iter, batch_size):

    for _ in range(batch_size):
        current = sess.run(iter)




def train_test_dataset(model_key, train, validate, test,
                       model_args=None, env=None):

    prepare_placeholder(train)

    iter = train.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        initial = sess.run([iter])
    exit()

    iter = train.make_one_shot_iterator()

    X, y = iter.get_next()
    logit_pred = stack_rank_model(X, y.shape[1])

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_regularization_loss = tf.reduce_sum(graph_regularizers)

    with tf.name_scope("cost_function") as scope:
        epsilon = 0.0001
        logits_ = tf.clip_by_value(logit_pred, epsilon, 1-epsilon)
        logits_ = tf.log(logits_ / (1 - logits_))
        cross_entropy = loss_function(
            y, logit_pred
        )

        cost = tf.reduce_mean(cross_entropy) + total_regularization_loss
        tf.summary.scalar("mean_entropy", cost)

    optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)
    merge_op = tf.summary.merge_all()

    time_string = datetime.datetime.now().isoformat()

    name = f"MLP_Test_{time_string}"
    avg_loss = []

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        summary_writer = tf.summary.FileWriter(os.path.join(env.get_cache_dir(), name), graph=sess.graph)

        i = 0

        try:
            while True:
                #print(sess.run(tf.stack([y_pred, cross_y, tf.cast(correct_prediction, tf.float64)], axis=1)))
                _, loss = sess.run(
                            [optimizer, cost])
                avg_loss.append(loss)

                print("Iteration %i Loss %f" % (i, loss))


                if i % 10 == 0:
                    loss = np.mean(avg_loss)
                    # print("Iteration %i Loss %f Acc %f Test_Loss %f Test_Acc %f" % (i, loss, acc, test_loss, test))
                    avg_loss = []

                summary_str = sess.run(merge_op)
                summary_writer.add_summary(summary_str, i)
                i += 1
        except tf.errors.OutOfRangeError:
            pass
