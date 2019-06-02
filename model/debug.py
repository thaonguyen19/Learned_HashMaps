from dataloader import *
import tensorflow as tf
import numpy as np
import argparse
import math

VAL_RATIO = 0.2
TEST_RATIO = 0.2
LOAD_FACTOR = 0.5

def train(args):
    # data_set = load_synthetic_data(args.data_dir, args.norm_label)
    # data_sets = create_train_validate_test_data_sets(data_set, VAL_RATIO, TEST_RATIO)
    train_dataset = load_shuttle_data(args.data_dir + '.trn', args.norm_label)
    validation_dataset = load_shuttle_data(args.data_dir + '.tst', args.norm_label)
    data_sets = create_train_validate_data_sets(train_dataset, validation_dataset)
    max_step = data_sets.train.num_keys//args.batch_size

    keys_placeholder = tf.placeholder(tf.float64, shape=(None, data_sets.train.key_size), name="keys")
    labels_placeholder = tf.placeholder(tf.float64, shape=(None), name="labels")

    n_hidden = 32
    keys_norm = (keys_placeholder - data_sets.train.keys_mean)/data_sets.train.keys_std
    # W1 = tf.Variable(tf.truncated_normal([data_sets.train.key_size, n_hidden], stddev=1.0 / math.sqrt(float(data_sets.train.key_size + n_hidden)), dtype=tf.float64), dtype=tf.float64)
    # b1 = tf.Variable(tf.zeros([n_hidden], dtype=tf.float64), dtype=tf.float64)
    # W2 = tf.Variable(tf.truncated_normal([n_hidden, 1], stddev=1.0 / math.sqrt(float(n_hidden + 1)), dtype=tf.float64), dtype=tf.float64)
    # b2 = tf.Variable(tf.zeros([1], dtype=tf.float64), dtype=tf.float64)
    # h = tf.nn.relu(tf.matmul(keys_placeholder, W1) + b1)
    h1 = tf.layers.dense(inputs=keys_norm, units=n_hidden, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, units=n_hidden, activation=tf.nn.relu)
    preds = tf.layers.dense(inputs=h2, units=1)
    # preds = tf.matmul(h, W2) + b2

    loss = tf.losses.mean_squared_error(labels=labels_placeholder, predictions=preds)
    optimizer = tf.train.AdamOptimizer(args.lr)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(args.epoch):
            print("epoch: ", epoch)
            for step in range(max_step):
                keys_feed, labels_feed = data_sets.train.next_batch(args.batch_size, True)
                feed_dict = {keys_placeholder: keys_feed, labels_placeholder: labels_feed}
                _, thao = sess.run([train_op, loss], feed_dict=feed_dict)
                keyss, pred = sess.run([keys_norm, preds], feed_dict=feed_dict)
                print(keyss)
                if (step+1) == 1:
                    print('Step %d: loss = %.10f' % (step, np.sqrt(thao)))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-data_dir', default='../data/normal_mean=1_std=1.txt')
    parser.add_argument('-lr', type=float)
    parser.add_argument('-batch_size', type=int)
    parser.add_argument('-hidden_width', nargs='+', type=int, default=[16])
    parser.add_argument('-epoch', type=int, default=5)
    parser.add_argument('-norm_label', action='store_true', help='Whether to normalize labels to be within [0,1]')
    parser.add_argument('-fix_inputs', action='store_true', help='Whether to keep input distribution the same and avoid standardization')
    args = parser.parse_args()
    #inference(args)
    train(args)
