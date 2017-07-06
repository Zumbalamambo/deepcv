import functools

import tensorflow as tf

from model.base.vgg import vgg_arg_scope
from model.base.resnet_v1 import resnet_arg_scope
from model.base.inception import inception_v1_arg_scope, inception_v2_arg_scope, \
    inception_v3_arg_scope, inception_v4_arg_scope, inception_resnet_v2_arg_scope
from model.base.mobilenet_v1 import mobilenet_v1_arg_scope

import model.classification.inception as inception
import model.classification.mobilenet_v1 as mobilenet_v1
import model.classification.resnet_v1 as resnet_v1
import model.classification.vgg as vgg

slim = tf.contrib.slim

arg_scopes_map = {
    'vgg_a': vgg_arg_scope,
    'vgg_16': vgg_arg_scope,
    'vgg_19': vgg_arg_scope,
    'resnet_v1_50': resnet_arg_scope,
    'resnet_v1_101': resnet_arg_scope,
    'resnet_v1_152': resnet_arg_scope,
    'resnet_v1_200': resnet_arg_scope,
    'inception_v1': inception_v1_arg_scope,
    'inception_v2': inception_v2_arg_scope,
    'inception_v3': inception_v3_arg_scope,
    'inception_v4': inception_v4_arg_scope,
    'inception_resnet_v2': inception_resnet_v2_arg_scope,
    'mobilenet_v1': mobilenet_v1_arg_scope
}

networks_map = {
    'vgg_a': vgg.vgg_a,
    'vgg_16': vgg.vgg_16,
    'vgg_19': vgg.vgg_19,
    'resnet_v1_50': resnet_v1.resnet_v1_50,
    'resnet_v1_101': resnet_v1.resnet_v1_101,
    'resnet_v1_152': resnet_v1.resnet_v1_152,
    'resnet_v1_200': resnet_v1.resnet_v1_200,
    'inception_v1': inception.inception_v1,
    'inception_v2': inception.inception_v2,
    'inception_v3': inception.inception_v3,
    'inception_v4': inception.inception_v4,
    'inception_resnet_v2': inception.inception_resnet_v2,
    'mobilenet_v1': mobilenet_v1.mobilenet_v1
}


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
    if name not in networks_map:
        raise ValueError("Name of network is not known %s" % name)
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
