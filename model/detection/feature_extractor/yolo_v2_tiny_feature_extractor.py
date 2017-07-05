import inspect

import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils.tfnet as tfnet


def tiny(net, classes, num_anchors, training=False, center=True):
    def batch_norm(net):
        net = slim.batch_norm(net, center=center, scale=True, epsilon=1e-5, is_training=training)
        if not center:
            net = tf.nn.bias_add(net,slim.variable('biases',
                                                   shape=[tf.shape(net)[-1]],
                                                   initializer=tf.zeros_initializer()))
        return net

    scope = __name__.split('.')[-2] + '_' + inspect.stack()[0][3]
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        normalizer_fn=batch_norm,
                        activation_fn=tfnet.leaky_relu), \
                        slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):
        index = 0
        channels = 16
        for _ in range(5):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, stride=1, scope='%s/max_pool%d' % (scope, index))
        index += 1
        channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
    net = slim.layers.conv2d(net, num_anchors * (5 + classes), kernel_size=[1, 1], activation_fn=None,
                             scope='%s/conv' % scope)
    net = tf.identity(net, name='%s/output' % scope)
    return scope, net


def _tiny(net, classes, num_anchors, training=False):
    return tiny(net, classes, num_anchors, training, False)


def darknet(net, classes, num_anchors, training=False, center=True):
    def batch_norm(net):
        net = slim.batch_norm(net, center=center, scale=True, epsilon=1e-5, is_training=training)
        if not center:
            net = tf.nn.bias_add(net,
                                 slim.variable('biases', shape=[tf.shape(net)[-1]], initializer=tf.zeros_initializer()))
        return net

    scope = __name__.split('.')[-2] + '_' + inspect.stack()[0][3]
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope(
            [slim.layers.conv2d], kernel_size=[3, 3], normalizer_fn=batch_norm,
            activation_fn=tfnet.leaky_relu), \
            slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):

        index = 0
        channels = 32
        for _ in range(2):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        for _ in range(2):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        passthrough = tf.identity(net, name=scope + '/passthrough')
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        channels *= 2
        # downsampling finished
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        with tf.name_scope(scope):
            _net = tfnet.reorg(passthrough)
        net = tf.concat([_net, net], 3, name='%s/concat%d' % (scope, index))
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
    net = slim.layers.conv2d(net, num_anchors * (5 + classes), kernel_size=[1, 1], activation_fn=None,
                             scope='%s/conv' % scope)
    net = tf.identity(net, name='%s/output' % scope)
    return scope, net


def _darknet(net, classes, num_anchors, training=False):
    return darknet(net, classes, num_anchors, training, False)
