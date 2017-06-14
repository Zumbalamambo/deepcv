import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_labels(ax, names, width, height, cell_width, cell_height, mask, prob, coords, xy_min, xy_max, areas,
                rtol=1e-3):
    colors = [prop['color'] for _, prop in zip(names, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    plots = []
    for i, (_mask, _prob, _coords, _xy_min, _xy_max, _areas) in enumerate(
            zip(mask, prob, coords, xy_min, xy_max, areas)):
        _mask = _mask.reshape([])
        _coords = _coords.reshape([-1])
        if np.any(_mask) > 0:
            index = np.argmax(_prob)
            iy = i // cell_width
            ix = i % cell_width
            plots.append(ax.add_patch(
                patches.Rectangle((ix * width / cell_width, iy * height / cell_height), width / cell_width,
                                  height / cell_height, linewidth=0, facecolor=colors[index], alpha=.2)))
            # check coords
            offset_x, offset_y, _w_sqrt, _h_sqrt = _coords
            cell_x, cell_y = ix + offset_x, iy + offset_y
            x, y = cell_x * width / cell_width, cell_y * height / cell_height
            _w, _h = _w_sqrt * _w_sqrt, _h_sqrt * _h_sqrt
            w, h = _w * width, _h * height
            x_min, y_min = x - w / 2, y - h / 2
            plots.append(ax.add_patch(
                patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor=colors[index], facecolor='none')))
            plots.append(ax.annotate(names[index], (x_min, y_min), color=colors[index]))
            # check offset_xy_min and xy_max
            wh = _xy_max - _xy_min
            assert np.all(wh >= 0)
            np.testing.assert_allclose(wh / [cell_width, cell_height], [[_w, _h]], rtol=rtol)
            np.testing.assert_allclose(_xy_min + wh / 2, [[offset_x, offset_y]], rtol=rtol)
    return plots
