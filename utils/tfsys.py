import os
import pickle
import sys


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        # detection = pickle.load(fp,encoding='iso-8859-1')
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data


def exists_or_mkdir(path, verbose=True):
    if not os.path.exists(path):
        if verbose:
            print("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            print("[!] %s exists ..." % path)
        return True


def load_config(config, path):
    # for path in paths:
    #     path = os.path.expanduser(os.path.expandvars(path))
    #     print(path)
    assert os.path.exists(path), 'No this config file'
    config.read(path)


def get_cachedir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('config', 'basedir')))
    return os.path.join(basedir, 'cache')


def get_dsdir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('config', 'basedir')))
    ds_name = config.get('dataset', 'name')
    return os.path.join(basedir, 'cache', 'dataset', ds_name)


def get_logdir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('config', 'basedir')))
    model = config.get('config', 'model')
    inference = config.get(model, 'inference')
    name = config.get('dataset', 'name')
    return os.path.join(basedir, 'cache', 'log', model, inference, name)


def get_label(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('config', 'basedir')))
    label_name = config.get('dataset', 'name')

    return os.path.join(basedir, 'config', 'label', label_name)
