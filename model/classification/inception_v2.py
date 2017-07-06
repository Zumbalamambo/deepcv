"""Contains the definition for inception v2 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model.base.inception_util as inception_util
import model.base.inception_v2 as inception_v2

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v2(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV2'):
    """Inception v2 model for classification.

    Constructs an Inception v2 network for classification as described in
    http://arxiv.org/abs/1502.03167.

    The default image size used to train this network is 224x224.

    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: the percentage of activation values that are retained.
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      prediction_fn: a function to get predictions out of logits.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, num_classes]
      end_points: a dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0
    """
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    # Final pooling and prediction
    with tf.variable_scope(scope, 'InceptionV2', [inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v2.inception_v2_base(inputs, scope=scope, min_depth=min_depth,
                                                depth_multiplier=depth_multiplier)
            with tf.variable_scope('Logits'):
                kernel_size = inception_v2.reduced_kernel_size_for_small_input(net, [7, 7])
                net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                      scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                # 1 x 1 x 1024
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                     scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


inception_v2.default_image_size = 224