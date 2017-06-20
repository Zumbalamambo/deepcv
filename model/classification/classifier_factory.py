import functools
import tensorflow as tf
import model.classification.inception as inception
import model.classification.mobilenet_v1 as mobilenet_v1

slim = tf.contrib.slim

networks_map = {
    'inception_v1': inception.inception_v1,
    'mobilenet_v1': mobilenet_v1.mobilenet_v1
}

arg_scopes_map = {
    'inception_v1': inception.inception_v1_arg_scope,
    'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope
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
