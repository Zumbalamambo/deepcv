import itertools

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import utils.tfdetection as tfdet
import utils.tfimage as tfimage
import utils.tfvisualize as tfvisualize


def detect_image(sess, model, names, image_placeholder, image_path, args):
    _, height, width, _ = image_placeholder.get_shape().as_list()

    _image = Image.open(image_path)
    image_original = np.array(np.uint8(_image))
    image_height, image_width, _ = image_original.shape
    image_std = tfimage.per_image_standardization(np.array(np.uint8(_image.resize((width, height)))).astype(np.float32))

    feed_dict = {image_placeholder: np.expand_dims(image_std, 0)}
    tensors = [model.conf, model.xy_min, model.xy_max]
    conf, xy_min, xy_max = sess.run([tf.check_numerics(t, t.op.name) for t in tensors], feed_dict=feed_dict)

    boxes = tfdet.non_max_suppress(conf[0], xy_min[0], xy_max[0], args.threshold, args.threshold_iou)
    scale = [image_width / model.cell_width, image_height / model.cell_height]

    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image_original)
    colors = [prop['color'] for _, prop in zip(names, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    cnt = 0
    for _conf, _xy_min, _xy_max in boxes:
        index = np.argmax(_conf)
        if _conf[index] > args.threshold:
            wh = _xy_max - _xy_min
            _xy_min = _xy_min * scale
            _wh = wh * scale
            linewidth = min(_conf[index] * 15, 3)
            ax.add_patch(patches.Rectangle(_xy_min, _wh[0], _wh[1], linewidth=linewidth, edgecolor=colors[index],
                                           facecolor='none'))
            ax.annotate(names[index] + '(%.1f%%)' % (_conf[index] * 100), _xy_min, color=colors[index])
            cnt += 1
    fig.canvas.set_window_title('%d objects detected' % cnt)
    ax.set_xticks([])
    ax.set_yticks([])


def detect_video(sess, model, names, image_placeholder, video_path, args):
    _, height, width, _ = image_placeholder.get_shape().as_list()

    video_capture = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame_bgr = video_capture.read()
            assert ret, 'Error video capture'
            image_height, image_width, _ = frame_bgr.shape
            scale = [image_width / model.cell_width, image_height / model.cell_height]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_std = np.expand_dims(
                tfimage.per_image_standardization(cv2.resize(frame_rgb, (width, height))).astype(np.float32), 0)

            feed_dict = {image_placeholder: frame_std}
            tensors = [model.conf, model.xy_min, model.xy_max]
            conf, xy_min, xy_max = sess.run([tf.check_numerics(t, t.op.name) for t in tensors], feed_dict=feed_dict)

            boxes = tfdet.non_max_suppress(conf[0], xy_min[0], xy_max[0], args.threshold, args.threshold_iou)

            for _conf, _xy_min, _xy_max in boxes:
                index = np.argmax(_conf)
                if _conf[index] > args.threshold:
                    _xy_min = (_xy_min * scale).astype(np.int)
                    _xy_max = (_xy_max * scale).astype(np.int)
                    cv2.rectangle(frame_bgr, tuple(_xy_min), tuple(_xy_max), (255, 0, 255), 3)
                    cv2.putText(frame_bgr, names[index] + '(%1.f%%)' % (_conf[index] * 100), tuple(_xy_min),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Video Detection', frame_bgr)
            cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()


class DetectImageManual(object):
    def __init__(self, sess, model, names, image, labels, cell_width, cell_height, feed_dict):
        self.sess = sess
        self.model = model
        self.names = names
        self.image = image
        self.labels = labels
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.feed_dict = feed_dict

        self.height, self.width, _ = image.shape
        self.scale = [self.width / self.cell_width, self.height / self.cell_height]

        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.ax.set_xticks(np.arange(0, self.width, self.width / cell_width))
        self.ax.set_yticks(np.arange(0, self.height, self.height / cell_height))
        self.ax.grid(which='both')
        self.ax.tick_params(labelbottom='off', labelleft='off')
        self.ax.imshow(image)

        self.plots = tfvisualize.draw_labels(self.ax, names, self.width, self.height, cell_width, cell_height, *labels)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.colors = [prop['color'] for _, prop in zip(names, itertools.cycle(plt.rcParams['axes.prop_cycle']))]

    def onclick(self, event):
        for p in self.plots:
            p.remove()
        self.plots = []

        height, width, _ = self.image.shape
        ix = int(event.xdata * self.cell_width / width)
        iy = int(event.ydata * self.cell_height / height)
        self.plots.append(self.ax.add_patch(
            patches.Rectangle((ix * width / self.cell_width, iy * height / self.cell_height),
                              width / self.cell_width, height / self.cell_height,
                              linewidth=0, facecolor='black', alpha=.2
                              )
        ))
        index = iy * self.cell_width + ix
        prob, iou, xy_min, wh = self.sess.run([self.model.prob[0][index], self.model.iou[0][index],
                                               self.model.xy_min[0][index], self.model.wh[0][index]],
                                              feed_dict=self.feed_dict)
        xy_min = xy_min * self.scale
        wh = wh * self.scale
        for _prob, _iou, (x, y), (w, h), color in zip(prob, iou, xy_min, wh, self.colors):
            index = np.argmax(_prob)
            name = self.names[index]
            _prob = _prob[index]
            _conf = _prob * _iou
            linewidth = min(_conf * 13, 3)
            self.plots.append(self.ax.add_patch(
                patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor=color, facecolor='none')
            ))
            self.plots.append(self.ax.annotate(name + '(%.1f%%, %.1f%%)' % (_iou * 100, _prob * 100),
                                               (x, y), color=color))
        self.fig.canvas.draw()
