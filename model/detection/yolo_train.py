import os
import configparser
import importlib
import shutil
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils.data
import utils.tfsys as tfsys
import utils.tftrain as tftrain
import utils.detection as util_det


def train(config, args):
    model = config.get('config', 'model')

    with open(tfsys.get_label(config), 'r') as f:
        labels = [line.strip() for line in f]

    width = config.getint(model, 'width')
    height = config.getint(model, 'height')
    cell_width, cell_height = util_det.calc_cell_width_height(config, width, height)
    tf.logging.warn('(width, height)=(%d, %d), (cell_width, cell_height)=(%d, %d)' % (width, height,
                                                                                      cell_width, cell_height))
    # prepare data batch
    paths = [os.path.join(tfsys.get_dsdir(config), profile + '.tfrecord') for profile in args.profile]
    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in paths)
    tf.logging.warn('num_examples=%d' % num_examples)
    with tf.name_scope('batch'):
        image_rgb, labels = utils.data.load_image_labels(paths, len(labels), width, height, cell_width, cell_height,
                                                         config)
        with tf.name_scope('per_image_standardization'):
            image_std = tf.image.per_image_standardization(image_rgb)
        batch = tf.train.shuffle_batch((image_std,) + labels, batch_size=args.batch_size,
                                       capacity=config.getint('queue', 'capacity'),
                                       min_after_dequeue=config.getint('queue', 'min_after_dequeue'),
                                       num_threads=multiprocessing.cpu_count()
                                       )
    # prepare model
    yolo = importlib.import_module('model.detection.'+model)
    builder = yolo.Builder(args, config)
    builder(batch[0], training=True)
    with tf.name_scope('total_loss') as name:
        builder.create_objectives(batch[1:])
        total_loss = tf.losses.get_total_loss(name=name)

    variables_to_restore = slim.get_variables_to_restore()
    global_step = tf.contrib.framework.get_or_create_global_step()
    with tf.name_scope('optimizer'):
        try:
            decay_steps = config.getint('exponential_decay', 'decay_steps')
            decay_rate = config.getfloat('exponential_decay', 'decay_rate')
            staircase = config.getboolean('exponential_decay', 'staircase')
            learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps, decay_rate,
                                                       staircase=staircase)
            tf.logging.warn('learning rate from %f with exponential decay (steps=%d, rate=%f, staircase=%d)' \
                            % (args.learning_rate, decay_steps, decay_rate, staircase))
        except (configparser.NoSectionError, configparser.NoOptionError):
            learning_rate = args.learning_rate
            tf.logging.warn('using a staionary learning rate %f' % args.learning_rate)

        optimizer = tftrain.get_optimizer(config, args.optimizer)(learning_rate)
        tf.logging.warn('optimizer=' + args.optimizer)
        train_op = slim.learning.create_train_op(total_loss, optimizer, global_step,
                                                 clip_gradient_norm=args.gradient_clip,
                                                 summarize_gradients=config.getboolean('summary', 'gradients'),
                                                 )

    # method1: fine tuning net weights from pre-trained net
    if args.transfer:
        path = os.path.expanduser(os.path.expandvars(args.transfer))
        tf.logging.warn('load checkpoint from ' + path)
        model_path = tf.train.latest_checkpoint(path)
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(model_path, variables_to_restore)

        def init_fn(sess):
            sess.run(init_assign_op, init_feed_dict)
            tf.logging.warn('loadding from global_step=%d, learning_rate=%f' % sess.run((global_step, learning_rate)))

    # method 2: training net weights from the beginning
    else:
        tf.logging.warn('start training from beginning')
        init_fn = lambda sess: tf.logging.warn('global_step=%d, learning_rate=%f' % sess.run((global_step, learning_rate)))

    tftrain.summary(config)
    logdir = tfsys.get_logdir(config)
    tf.logging.warn('tensorboard --logdir ' + logdir)

    slim.learning.train(train_op=train_op, logdir=logdir, global_step=global_step, number_of_steps=args.steps,
                        init_fn=init_fn, summary_writer=tf.summary.FileWriter(os.path.join(logdir, args.logname)),
                        save_summaries_secs=args.summary_secs, save_interval_secs=args.save_secs,
                        )