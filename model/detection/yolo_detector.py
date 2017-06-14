import itertools
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import utils.detection as util_det
import utils.detection.visualize


def detect_image(sess, model, names, image_placeholder, image_path):
    _, height, width, _ = image_placeholder.get_shape().as_list()

    _image = Image.open(image_path)
    image_original = np.array(np.uint8(_image))
    image_height, image_width, _ = image_original.shape
    image_std = util_det.preprocess.normalize_input(np.array(np.uint8(_image.resize((width, height)))).astype(np.float32))

    feed_dict = {image_placeholder: np.expand_dims(image_std, 0)}
    tensors = [model.conf, model.xy_min, model.xy_max]
    conf, xy_min, xy_max = sess.run([tf.check_numerics(t, t.op.name) for t in tensors], feed_dict=feed_dict)

    boxes = util_det.postprocess.non_max_suppress(conf[0], xy_min[0], xy_max[0], args.threshold, args.threshold_iou)
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


def detect_video(sess, model, names, image_placeholder, video_path):
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
                util_det.preprocess.normalize_input(cv2.resize(frame_rgb, (width, height))).astype(np.float32), 0)

            feed_dict = {image_placeholder: frame_std}
            tensors = [model.conf, model.xy_min, model.xy_max]
            conf, xy_min, xy_max = sess.run([tf.check_numerics(t, t.op.name) for t in tensors], feed_dict=feed_dict)

            boxes = util_det.postprocess.non_max_suppress(conf[0], xy_min[0], xy_max[0], args.threshold, args.threshold_iou)

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


class Drawer(object):
    def __init__(self, sess, model, names, image, labels, cell_width, cell_height, feed_dict):
        self.sess = sess
        self.model = model
        self.names = names
        self.images = image
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

        self.plots = utils.detection.visualize.draw_labels(self.ax, names, self.width, self.height, cell_width, cell_height,
                                                 *labels)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.colors = [prop['color'] for _, prop in zip(names, itertools.cycle(plt.rcParams['axes.prop_cycle']))]

    def onclick(self, event):
        for p in self.plots:
            p.remove()
        self.plots = []
        ix = int(event.xdata * self.cell_width / self.width)
        iy = int(event.ydata * self.cell_height / self.height)
        self.plots.append(self.ax.add_patch(
            patches.Rectangle((ix * self.width / self.cell_width, iy * self.height / self.cell_height),
                              self.width / self.cell_width, self.height / self.cell_height,
                              linewidth=0, facecolor='black', alpha=.2
                              )
        ))
        index = iy * self.cell_width + ix
        prob, iou, xy_min, wh = self.sess.run([self.model.prob[0][index], self.model.iou[0][index]])
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
            self.plots.append(self.ax.add_patch(
                self.ax.annotate(name + '(%.1f%%, %.1f%%)' % (_iou * 100, _prob * 100), (x, y), color=color)
            ))
        self.fig.canvas.draw()

#
# def run():
#     model = config.get('config', 'model')
#     yolo = importlib.import_module(model)
#
#     width = config.getint(model, 'width')
#     height = config.getint(model, 'height')
#
#     cell_width, cell_height = det_util.calc_cell_width_height(config, width, height)
#
#
#     with open(os.path.join(cachedir, 'names')) as f:
#         names = [line.strip() for line in f]
#
#     file_path = os.path.expanduser(os.path.expandvars(args.path))
#     ext_name = os.path.splitext(os.path.basename(file_path))[1]
#
#     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#
#         if ext_name == '.tfrecord':
#
#             num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(file_path))
#             tf.logging.warn('num_examples=%d' % num_examples)
#             file_path = [file_path]
#             image_rgb, labels = det_util.data.load_image_labels(file_path, len(names), width, height, cell_width,
#                                                                 cell_height, config)
#             image_std = tf.image.per_image_standardization(image_rgb)
#             image_rgb = tf.cast(image_rgb, tf.uint8)
#
#             image_placeholder = tf.placeholder(image_std.dtype, [1] + image_std.get_shape().as_list(),
#                                                name='image_placeholder')
#             label_placeholder = [tf.placeholder(l.dtype, [1] + l.get_shape().as_list(), name=l.op.name + '_placeholder')
#                                  for l in labels]
#
#             builder = yolo.Builder(args, config)
#             builder(image_placeholder)
#             with tf.name_scope('total_loss') as name:
#                 builder.create_objectives(label_placeholder)
#                 total_loss = tf.losses.get_total_loss(name=name)
#
#             global_step = tf.contrib.framework.get_or_create_global_step()
#             restore_variables = slim.get_variables_to_restore()
#             tf.global_variables_initializer().run()
#
#             coord = tf.train.Coordinator()
#             threads = tf.train.start_queue_runners(sess, coord)
#             _image_rgb, _image_std, _labels = sess.run([image_rgb, image_std, labels])
#             coord.request_stop()
#             coord.join(threads)
#
#             model_path = tf.train.latest_checkpoint(utils.get_logdir(config))
#             slim.assign_from_checkpoint_fn(model_path, restore_variables)(sess)
#
#             feed_dict = dict([(ph, np.expand_dims(d, 0)) for ph, d in zip(label_placeholder, _labels)])
#             feed_dict[image_placeholder] = np.expand_dims(_image_std, 0)
#
#             _ = Drawer(sess, builder.model, builder.names, _image_rgb, _labels,
#                        builder.model.cell_width, builder.model.cell_height, feed_dict)
#             plt.show()
#
#         else:
#             image_placeholder = tf.placeholder(tf.float32, [1, height, width, 3], name='image')
#
#             builder = yolo.Builder(args, config)
#             builder(image_placeholder)
#
#             global_step = tf.contrib.framework.get_or_create_global_step()
#
#             model_path = tf.train.latest_checkpoint(utils.get_logdir(config))
#             slim.assign_from_checkpoint_fn(model_path, tf.global_variables())(sess)
#
#             tf.logging.info('global_step=%d' % sess.run(global_step))
#
#             if os.path.isfile(file_path):
#                 if ext_name in ['.jpg', '.png']:
#                     detect_image(sess, builder.model, builder.names, image_placeholder, file_path)
#                     plt.show()
#                 elif ext_name in ['.avi', '.mp4']:
#                     detect_video(sess, builder.model, builder.names, image_placeholder, file_path)
#                 else:
#                     print('No this file type')
#             else:
#                 pass
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('path', help='input image path')
#     parser.add_argument('--type', default='image', help='file type')
#     parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
#     parser.add_argument('-p', '--preprocess', default='std', help='the preprocess function')
#     parser.add_argument('-t', '--threshold', type=float, default=0.3)
#     parser.add_argument('--threshold_iou', type=float, default=0.4, help='IoU threshold')
#     parser.add_argument('-e', '--exts', nargs='+', default=['.jpg', '.png'])
#     parser.add_argument('-m', '--manual', type=bool, default=False, help='')
#     parser.add_argument('--level', default='info', help='logging level')
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     args = main()
#     config = configparser.ConfigParser()
#     print(args.config)
#     utils.load_config(config, args.config)
#     if args.level:
#         tf.logging.set_verbosity(args.level.upper())
#     run()
