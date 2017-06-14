import os
import re
import importlib
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_cachedir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('config', 'basedir')))
    name = os.path.basename(config.get('cache', 'names'))
    return os.path.join(basedir, 'dataset', name)


def get_logdir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('config', 'basedir')))
    model = config.get('config', 'model')
    inference = config.get(model, 'inference')
    name = os.path.basename(config.get('cache', 'names'))
    return os.path.join(basedir, 'log', model, inference, name)


def get_inference(config):
    model = config.get('config', 'model')
    return getattr(importlib.import_module('.'.join([model, 'inference'])), config.get(model, 'inference'))


def get_downsampling(config):
    model = config.get('config', 'model')
    return getattr(importlib.import_module('.'.join([model, 'inference'])),
                   config.get(model, 'inference').upper() + '_DOWNSAMPLING')


def calc_cell_width_height(config, width, height):
    downsampling_width, downsampling_height = get_downsampling(config)
    assert width % downsampling_width == 0
    assert height % downsampling_height == 0
    return width // downsampling_width, height // downsampling_height


def match_trainable_variables(pattern):
    prog = re.compile(pattern)
    return [v for v in tf.trainable_variables() if prog.match(v.op.name)]


def match_tensor(pattern):
    prog = re.compile(pattern)
    return [op.values()[0] for op in tf.get_default_graph().get_operations() if op.values() and prog.match(op.name)]


def load_config(config, paths):
    for path in paths:
        path = os.path.expanduser(os.path.expandvars(path))
        assert os.path.exists(path)
        config.read(path)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
