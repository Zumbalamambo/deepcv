import tensorflow as tf


def leaky_relu(inputs, alpha=.1):
    with tf.name_scope('leaky_relu') as name:
        data = tf.identity(inputs, name='data')

    return tf.maximum(data, alpha * data, name=name)


def reorg(net, stride=2, name='reorg'):
    batch_size, height, width, channels = net.get_shape().as_list()
    _height, _width, _channel = height // stride, width // stride, channels * stride * stride
    with tf.name_scope(name) as name:
        net = tf.reshape(net, [batch_size, _height, stride, _width, stride, channels])
        net = tf.transpose(net, [0, 1, 3, 2, 4, 5]) # batch_size, _height, _width, stride, stride, channels
        net = tf.reshape(net, [batch_size, _height, _width, -1], name=name)

    return net