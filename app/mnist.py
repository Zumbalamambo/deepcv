#! /usr/bin/python
# -*- coding: utf8 -*-

import sys, time

sys.path.append('..')
import tensorflow as tf
import utils.tfdata as tfdata
import utils.tflayer as tflayer
import utils.tftrain as tftrain


def run(args):
    if args.task == 'classify':
        train_test_val_mnist_v1(args.model, args.gpu)
    else:
        print("No this task")


def train_test_val_mnist_v1(model, gpu):
    X_train, y_train, X_val, y_val, X_test, y_test = tfdata.load_mnist_dataset(shape=(-1, 784))

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    # net
    net = tflayer.InputLayer(x, name='input')

    net = tflayer.DenseLayer(net, out_units=800, act=tf.nn.relu, name='relu1')
    net = tflayer.DropoutLayer(net, keep=0.5, name='drop2')
    net = tflayer.DenseLayer(net, out_units=800, act=tf.nn.relu, name='relu2')
    net = tflayer.DropoutLayer(net, keep=0.5, name='drop3')
    net = tflayer.DenseLayer(net, out_units=10, act=tf.identity, name='output')

    y = net.outputs

    net.print_params()
    net.print_layers()

    # training
    learning_rate = 0.0001
    cost = tftrain.cross_entropy(y, y_, name='sigmoid_cross_entropy')
    train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                        epsilon=1e-08, use_locking=False).minimize(cost)

    sess.run(tf.global_variables_initializer())
    # start training
    n_epoch = 100
    n_steps = 1000
    batch_size = 128
    print_freq = 10

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for epoch in range(n_epoch):
        start_time = time.time()
        # n = 1
        for X_train_a, y_train_a in tfdata.minibatches(X_train, y_train,
                                                       batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)
            if print_freq % 10 == 0:
                train_acc = acc.eval(feed_dict=feed_dict)
                print(train_acc)
            train_step.run(feed_dict=feed_dict)


def main():
    train_test_val_mnist_v1()


if __name__ == '__main__':
    main()
