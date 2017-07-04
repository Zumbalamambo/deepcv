import importlib

import numpy as np


def iou(xy_min1, xy_max1, xy_min2, xy_max2):
    assert (not np.isnan(xy_min1).any())
    assert (not np.isnan(xy_max1).any())
    assert (not np.isnan(xy_min2).any())
    assert (not np.isnan(xy_max2).any())
    assert np.all(xy_min1 <= xy_max1)
    assert np.all(xy_min2 <= xy_max2)
    areas1 = np.multiply.reduce(xy_max1 - xy_min1)
    areas2 = np.multiply.reduce(xy_max2 - xy_min2)
    _xy_min = np.maximum(xy_min1, xy_min2)
    _xy_max = np.minimum(xy_max1, xy_max2)
    _wh = np.maximum(_xy_max - _xy_min, 0)
    _areas = np.multiply.reduce(_wh)
    assert _areas <= areas1
    assert _areas <= areas2
    return _areas / np.maximum(areas1 + areas2 - _areas, 1e-10)


def non_max_suppress(conf, xy_min, xy_max, threshold, threshold_iou):
    _, _, classes = conf.shape
    boxes = [(_conf, _xy_min, _xy_max) for _conf, _xy_min, _xy_max in
             zip(conf.reshape(-1, classes), xy_min.reshape(-1, 2), xy_max.reshape(-1, 2))]
    for c in range(classes):
        boxes.sort(key=lambda box: box[0][c], reverse=True)
        for i in range(len(boxes) - 1):
            box = boxes[i]
            if box[0][c] <= threshold:
                continue
            for _box in boxes[i + 1:]:
                if iou(box[1], box[2], _box[1], _box[2]) >= threshold_iou:
                    _box[0][c] = 0
    return boxes


def get_downsampling(config):
    model = config.get('config', 'model')
    # m_infer = 'model.detection.'+model+'.inference'
    print('name: ' + model)
    return getattr(importlib.import_module('model.detection.' + model + '.inference'),
                   config.get(model, 'inference').upper() + '_DOWNSAMPLING')


def calc_cell_width_height(config, width, height):
    downsampling_width, downsampling_height = get_downsampling(config)
    assert width % downsampling_width == 0
    assert height % downsampling_height == 0
    return width // downsampling_width, height // downsampling_height
