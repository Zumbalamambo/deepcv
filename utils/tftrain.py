import configparser
import inspect

import numpy as np
import tensorflow as tf

import detection.tfnet as tfnet


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets), 'the length of X,y is not same'
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def initialize_global_variables(sess=None):
    assert sess is not None
    sess.run(tf.global_variables_initializer())


def cross_entropy(logits, target, name=None, method=tf.nn.sparse_softmax_cross_entropy_with_logits):
    try:
        return tf.reduce_mean(method(logits=logits, targets=target, name=name))
    except:
        return tf.reduce_mean(method(logits=logits, labels=target, name=name))


def summary_scalar(config):
    try:
        reduce = eval(config.get('summary', 'scalar_reduce'))
        for t in tfnet.match_tensor(config.get('summary', 'scalar')):
            name = t.op.name
            if len(t.get_shape()) > 0:
                t = reduce(t)
                tf.logging.warn(name + ' is not a scalar tensor, reducing by ' + reduce.__name__)
            tf.summary.scalar(name, t)
    except (configparser.NoSectionError, configparser.NoOptionError):
        tf.logging.warn(inspect.stack()[0][3] + ' disabled')


def summary_image(config):
    try:
        for t in tfnet.match_tensor(config.get('summary', 'image')):
            name = t.op.name
            channels = t.get_shape()[-1].value
            if channels not in (1, 3, 4):
                t = tf.expand_dims(tf.reduce_sum(t, -1), -1)
            tf.summary.image(name, t, config.getint('summary', 'image_max'))
    except (configparser.NoSectionError, configparser.NoOptionError):
        tf.logging.warn(inspect.stack()[0][3] + ' disabled')


def summary_histogram(config):
    try:
        for t in tfnet.match_tensor(config.get('summary', 'histogram')):
            tf.summary.histogram(t.op.name, t)
    except (configparser.NoSectionError, configparser.NoOptionError):
        tf.logging.warn(inspect.stack()[0][3] + ' disabled')


def summary(config):
    summary_scalar(config)
    summary_image(config)
    summary_histogram(config)


def get_optimizer(config, name):
    '''
    If data is sparse, recommend adaptive method like Adagrad, RMSprop, Adam.
    Adagrad, RMSprop and Adam are similar at most time.
    Adam add bias-correction and momentum based on RMSprop,
    and while the gradient becomes sparse, the Adam will be better.

    :param config:
    :param name:
    :return:
    '''
    section = 'optimizer_' + name
    optimizer = {
        'gd': lambda learning_rate: tf.train.GradientDescentOptimizer(learning_rate),

        'adam': lambda learning_rate: tf.train.AdamOptimizer(learning_rate,
                                                             config.getfloat(section, 'beta1'),
                                                             config.getfloat(section, 'beta2'),
                                                             config.getfloat(section, 'epsilon')),

        'adadelta': lambda learning_rate: tf.train.AdadeltaOptimizer(learning_rate,
                                                                     config.getfloat(section, 'rho'),
                                                                     config.getfloat(section, 'epsilon')),

        'adagrad': lambda learning_rate: tf.train.AdagradOptimizer(learning_rate,
                                                                   config.getfloat(section, 'initial_accumulator_value')
                                                                   ),

        'momentum': lambda learning_rate: tf.train.MomentumOptimizer(learning_rate,
                                                                     config.getfloat(section, 'momentum')),

        'rmsprop': lambda learning_rate: tf.train.RMSPropOptimizer(learning_rate,
                                                                   config.getfloat(section, 'decay'),
                                                                   config.getfloat(section, 'momentum'),
                                                                   config.getfloat(section, 'epsilon')),

        'ftrl': lambda learning_rate: tf.train.FtrlOptimizer(learning_rate,
                                                             config.getfloat(section, 'learning_rate_power'),
                                                             config.getfloat(section, 'initial_accumulator_value'),
                                                             config.getfloat(section, 'l1_regularization_strength'),
                                                             config.getfloat(section, 'l2_regularization_strength')),
    }[name]

    return optimizer
