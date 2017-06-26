import tensorflow as tf

slim = tf.contrib.slim


def run(config, args):
    if args.task == 'train':
        pass
    elif args.task == 'predict':
        predict(config)


def predict(config):
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
