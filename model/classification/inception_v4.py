"""Contains the definition of the Inception V4 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model.base.inception_util as inception_util
from model.base.inception_v4 import inception_v4_base
slim = tf.contrib.slim


def inception_v4(inputs, num_classes=1001, is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='InceptionV4',
                 create_aux_logits=True):
    """Creates the Inception V4 model.

    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
      create_aux_logits: Whether to include the auxiliary logits.

    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
    with tf.variable_scope(scope, 'InceptionV4', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v4_base(inputs, scope=scope)

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # Auxiliary Head logits
                if create_aux_logits:
                    with tf.variable_scope('AuxLogits'):
                        # 17 x 17 x 1024
                        aux_logits = end_points['Mixed_6h']
                        aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID',
                                                     scope='AvgPool_1a_5x5')
                        aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1b_1x1')
                        aux_logits = slim.conv2d(aux_logits, 768, aux_logits.get_shape()[1:3], padding='VALID',
                                                 scope='Conv2d_2a')
                        aux_logits = slim.flatten(aux_logits)
                        aux_logits = slim.fully_connected(aux_logits, num_classes, activation_fn=None,
                                                          scope='Aux_logits')
                        end_points['AuxLogits'] = aux_logits

                # Final pooling and prediction
                with tf.variable_scope('Logits'):
                    # 8 x 8 x 1536
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a')
                    # 1 x 1 x 1536
                    net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
                    net = slim.flatten(net, scope='PreLogitsFlatten')
                    end_points['PreLogitsFlatten'] = net
                    # 1536
                    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                                  scope='Logits')
                    end_points['Logits'] = logits
                    end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
        return logits, end_points
