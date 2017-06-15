import importlib


def get_downsampling(config):
    model = config.get('config', 'model')
    # m_infer = 'model.detection.'+model+'.inference'
    print('name: '+model)
    return getattr(importlib.import_module('model.detection.'+model+'.inference'),
                   config.get(model, 'inference').upper() + '_DOWNSAMPLING')


def calc_cell_width_height(config, width, height):
    downsampling_width, downsampling_height = get_downsampling(config)
    assert width % downsampling_width == 0
    assert height % downsampling_height == 0
    return width // downsampling_width, height // downsampling_height