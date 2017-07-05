import re
import tensorflow as tf


def match_tensor(pattern):
    prog = re.compile(pattern)
    return [op.values()[0] for op in tf.get_default_graph().get_operations() if op.values() and prog.match(op.name)]


def match_trainable_variables(pattern):
    prog = re.compile(pattern)
    return [v for v in tf.trainable_variables() if prog.match(v.op.name)]