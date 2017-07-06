import inspect
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model.base.darknet_util as darknet_util


def tiny(net, classes, boxes_per_cell, training=False):
    scope = __name__.split('.')[-2] + '_' + inspect.stack()[0][3]
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], activation_fn=darknet_util.leaky_relu), \
         slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):
        index = 0
        net = slim.layers.conv2d(net, 16, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 32, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 64, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 128, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 256, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 512, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 512, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 1024, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 256, scope='%s/conv%d' % (scope, index))
    net = tf.identity(net, name='%s/conv' % scope)
    _, cell_height, cell_width, _ = net.get_shape().as_list()
    net = slim.layers.flatten(net, scope='%s/flatten' % scope)
    with slim.arg_scope([slim.layers.fully_connected], activation_fn=darknet_util.leaky_relu, weights_regularizer=slim.l2_regularizer(0.001)), \
         slim.arg_scope([slim.layers.dropout], keep_prob=.5, is_training=training):
        index = 0
        net = slim.layers.fully_connected(net, 256, scope='%s/fc%d' % (scope, index))
        net = slim.layers.dropout(net, scope='%s/dropout%d' % (scope, index))
        index += 1
        net = slim.layers.fully_connected(net, 4096, scope='%s/fc%d' % (scope, index))
        net = slim.layers.dropout(net, scope='%s/dropout%d' % (scope, index))
    net = slim.layers.fully_connected(net, cell_width * cell_height * (classes + boxes_per_cell * 5), activation_fn=None, scope='%s/fc' % scope)
    net = tf.identity(net, name='%s/output' % scope)
    return scope, net

TINY_DOWNSAMPLING = (2 ** 6, 2 ** 6)
