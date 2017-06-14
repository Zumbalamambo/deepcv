import os


def load_config(config, paths):
    for path in paths:
        path = os.path.expanduser(os.path.expandvars(path))
        assert os.path.exists(path), 'No this config file'
        config.read(path)


def get_cachedir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('config', 'basedir')))
    return os.path.join(basedir, 'cache')


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


