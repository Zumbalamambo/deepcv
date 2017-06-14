import numpy as np
import tensorflow as tf


def initialize_global_variables(sess=None):
    assert sess is not None
    sess.run(tf.global_variables_initializer())


def cross_entropy(logits, target, name=None, method=tf.nn.sparse_softmax_cross_entropy_with_logits):
    try:
        return tf.reduce_mean(method(logits=logits, targets=target, name=name))
    except:
        return tf.reduce_mean(method(logits=logits, labels=target, name=name))
