import tensorflow as tf
import numpy as numpy
import argparse
from dataloader import *

NUM_EXPERTS = 2

def train(args):
    train_dataset = load_shuttle_data(args.data_dir + '.trn', args.norm_label, NUM_EXPERTS)
    validation_dataset = load_shuttle_data(args.data_dir + '.tst', args.norm_label, NUM_EXPERTS)
    data_sets = create_train_validate_data_sets(train_dataset, validation_dataset)
    max_step = data_sets.train.num_keys//args.batch_size

    lr = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.int32, [None])

    X = (X - data_sets.train.keys_mean)/data_sets.train.keys_std

    h1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, units=512, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, units=256, activation=tf.nn.relu)
    pred = tf.layers.dense(inputs=h3, units=NUM_EXPERTS)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

    optimizer = tf.train.AdamOptimizer(lr)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(args.epoch):
        # print("epoch: ", epoch)
        total_loss, acc = 0, 0
        for step in range(max_step):
            keys_feed, labels_feed = data_sets.train.next_batch(args.batch_size, True)
            # print(keys_feed, labels_feed)
            feed_dict = {X: keys_feed, y: labels_feed, lr: args.lr}
            _, predictions, thao = sess.run([train_op, pred, loss], feed_dict=feed_dict)
            # print(predictions)
            total_loss += thao
            acc += np.sum(np.argmax(predictions, axis=1) == labels_feed)
        print('Epoch %d: loss = %.10f acc = %.10f' % (epoch, total_loss/max_step, acc/data_sets.train.num_keys))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-data_dir', default='../data/shuttle')
    parser.add_argument('-lr', type=float)
    parser.add_argument('-batch_size', type=int)
    parser.add_argument('-hidden_width', nargs='+', type=int, default=[16])
    parser.add_argument('-epoch', type=int, default=5)
    parser.add_argument('-norm_label', action='store_true', help='Whether to normalize labels to be within [0,1]')
    parser.add_argument('-fix_inputs', action='store_true', help='Whether to keep input distribution the same and avoid standardization')
    args = parser.parse_args()
    train(args)