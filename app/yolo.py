import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import model.detection.train_detector as train_detector
import model.detection.yolo_detector as yolo_det
import utils.tfdata as tfdata
import utils.tfdetection as tfdet
import utils.tfsys as tfsys

sys.path.append('..')


def run(config, args):
    if args.task == 'train':
        train(config, args)
    elif args.task == 'detect':
        detect(config, args)


def train(config, args):
    train_detector.run(config, args)


def detect(config, args):
    model = config.get('config', 'model')

    width = config.getint(model, 'width')
    height = config.getint(model, 'height')

    cell_width, cell_height = tfdet.calc_cell_width_height(config, width, height)

    with open(tfsys.get_label(config)) as f:
        names = [line.strip() for line in f]

    file_path = os.path.expanduser(os.path.expandvars(args.file))
    ext_name = os.path.splitext(os.path.basename(file_path))[1]

    yolo = importlib.import_module('model.detection.' + model)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        if ext_name == '.tfrecord':

            num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(file_path))
            tf.logging.warn('num_examples=%d' % num_examples)
            file_path = [file_path]
            image_rgb, labels = tfdata.load_image_labels(file_path, len(names), width, height, cell_width,
                                                         cell_height, config)
            image_std = tf.image.per_image_standardization(image_rgb)
            image_rgb = tf.cast(image_rgb, tf.uint8)

            image_placeholder = tf.placeholder(image_std.dtype, [1] + image_std.get_shape().as_list(),
                                               name='image_placeholder')
            label_placeholder = [tf.placeholder(l.dtype, [1] + l.get_shape().as_list(), name=l.op.name + '_placeholder')
                                 for l in labels]

            builder = yolo.Builder(args, config)
            builder(image_placeholder)

            with tf.name_scope('total_loss') as name:
                builder.create_objectives(label_placeholder)
                total_loss = tf.losses.get_total_loss(name=name)

            global_step = tf.contrib.framework.get_or_create_global_step()
            restore_variables = slim.get_variables_to_restore()
            tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            _image_rgb, _image_std, _labels = sess.run([image_rgb, image_std, labels])
            coord.request_stop()
            coord.join(threads)

            model_path = tf.train.latest_checkpoint(tfsys.get_logdir(config))
            slim.assign_from_checkpoint_fn(model_path, restore_variables)(sess)

            feed_dict = dict([(ph, np.expand_dims(d, 0)) for ph, d in zip(label_placeholder, _labels)])
            feed_dict[image_placeholder] = np.expand_dims(_image_std, 0)

            _ = yolo_det.DetectImageManual(sess, builder.model, builder.labels, _image_rgb, _labels,
                                           builder.model.cell_width, builder.model.cell_height, feed_dict)
            plt.show()

        else:
            image_placeholder = tf.placeholder(tf.float32, [1, height, width, 3], name='image')

            builder = yolo.Builder(args, config)
            builder(image_placeholder)

            global_step = tf.contrib.framework.get_or_create_global_step()

            model_path = tf.train.latest_checkpoint(tfsys.get_logdir(config))
            slim.assign_from_checkpoint_fn(model_path, tf.global_variables())(sess)

            tf.logging.info('global_step=%d' % sess.run(global_step))

            if os.path.isfile(file_path):
                if ext_name in ['.jpg', '.png']:
                    yolo_det.detect_image(sess, builder.model, builder.labels, image_placeholder, file_path, args)
                    plt.show()
                elif ext_name in ['.avi', '.mp4']:
                    yolo_det.detect_video(sess, builder.model, builder.labels, image_placeholder, file_path, args)
                else:
                    print('No this file type')
            else:
                pass
