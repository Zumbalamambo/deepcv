import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

import model.classification.classifier_factory as classfier_factory
import utils.dataset.imagenet as imagenet
import utils.preprocessing.preprocessing_factory as preprocessing_factory

try:
    import urllib2
except ImportError:
    import urllib.request as urllib

slim = tf.contrib.slim


def run(config, args):
    if args.task == 'classify':
        if args.file is not None:
            classify_image_local(config, args)
        elif args.file_url is not None:
            classify_iamge_remote(config, args)
        else:
            print('Classification error!')
    elif args.task == 'train':
        pass
    # elif args.task == 'test':
    #     test()
    else:
        print('No this task!')


def classify_image_local(config, args):
    # parse arguments
    model_name = config.get('config', 'model')
    num_class = int(config.get('dataset', 'num_class'))
    img_height = int(config.get(model_name, 'height'))
    img_width = int(config.get(model_name, 'width'))
    ckpt_path = config.get('weights', 'ckpt')
    model_variables = config.get('weights', 'model_variables')
    image_path = args.file

    with tf.Graph().as_default():
        # load file
        image_ph = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
        image_original = Image.open(image_path)
        image = image_original.resize((img_height, img_width))
        image = np.expand_dims(np.array(np.uint8(image)), 0)

        feed_dict = {image_ph: image}

        net_fn = classfier_factory.get_network_fn(model_name, num_class, is_training=False)
        logits, _ = net_fn(image_ph)
        net_out = tf.nn.softmax(logits)

        if args.gpu:
            sess_cfg = tf.ConfigProto()
            sess_cfg.gpu_options.allow_growth = True
            sess = tf.Session(config=sess_cfg)
        else:
            sess = tf.Session()

        with sess:
            if os.path.isdir(ckpt_path):
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(ckpt_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_model_variables(model_variables))
                init_fn(sess)
            classify_result = sess.run(net_out, feed_dict=feed_dict)
            probablities = classify_result[0, 0:]
            sorted_id = [i[0] for i in sorted(enumerate(-probablities), key=lambda x: x[1])]

        names = imagenet.create_readable_names_for_imagenet_labels()
        top5_result = []
        for i in range(5):
            index = sorted_id[i]
            top5_result.append({'name': names[index + 1], 'probability': probablities[index] * 100})
            if args.terminal:
                print('Probablity %.2f%% ==> [%s]' % ((probablities[index] * 100), names[index + 1]))

        if args.json:
            top5_result = json.dumps(top5_result)

        if args.show:
            pass

        sess.close()

    return top5_result


def classify_iamge_remote(config, args):
    # parse arguments
    model_name = config.get('config', 'model')
    # ds_name = config.get('dataset', 'name')
    num_class = int(config.get('dataset', 'num_class'))
    img_height = int(config.get(model_name, 'height'))
    img_width = int(config.get(model_name, 'width'))
    ckpt_path = config.get('weights', 'ckpt')
    model_variables = config.get('weights', 'model_variables')
    image_url = args.file_url

    with tf.Graph().as_default():
        image_str = urllib.urlopen(image_url).read()
        image = tf.image.decode_jpeg(image_str, channels=3)

        preprocess_image_fn = preprocessing_factory.get_preprocessing(model_name, is_training=True)

        processed_image = preprocess_image_fn(image, img_height, img_width)
        processed_image = tf.expand_dims(processed_image, 0)

        net_fn = classfier_factory.get_network_fn(model_name, num_class, is_training=False)
        logits, _ = net_fn(processed_image)
        net_out = tf.nn.softmax(logits)

        if args.gpu:
            sess_cfg = tf.ConfigProto()
            sess_cfg.gpu_options.allow_growth = True
            sess = tf.Session(config=sess_cfg)
        else:
            sess = tf.Session()

        with sess:
            if os.path.isdir(ckpt_path):
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(ckpt_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_model_variables(model_variables))
                init_fn(sess)

            np_image, probablities = sess.run([image, net_out])
            probablities = probablities[0, 0:]
            sorted_id = [i[0] for i in sorted(enumerate(-probablities), key=lambda x: x[1])]

        top5_result = []
        names = imagenet.create_readable_names_for_imagenet_labels()
        for i in range(5):
            index = sorted_id[i]
            top5_result.append({'name': names[index + 1], 'probability': probablities[index] * 100})
            if args.terminal:
                print('Probablity %.2f%% ==> [%s]' % ((probablities[index] * 100), names[index + 1]))

        if args.json:
            top5_result = json.dumps(top5_result)

        if args.show:
            pass
        sess.close()

    return top5_result

#
# def test():
#     image_size = vgg.vgg_16.default_image_size
#     with tf.device('/gpu:0'):
#         url = 'https://upload.wikimedia.org/wikipedia/commons/d/d9/First_Student_IC_school_bus_202076.jpg'
#         image_string = urllib.urlopen(url).read()
#         image = tf.image.decode_jpeg(image_string, channels=3)
#         preprocess_image_fn = preprocessing_factory.get_preprocessing('vgg_16', is_training=True)
#
#         processed_image = preprocess_image_fn(image, image_size, image_size)
#
#         processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
#         processed_images = tf.expand_dims(processed_image, 0)
#
#         # Create the model, use the default arg scope to configure the batch norm parameters.
#         with slim.arg_scope(vgg.vgg_arg_scope()):
#             # 1000 classes instead of 1001.
#             logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
#         probabilities = tf.nn.softmax(logits)
#
#         init_fn = slim.assign_from_checkpoint_fn(
#             os.path.join('cache/weight/vgg', 'vgg_16.ckpt'),
#             slim.get_model_variables('vgg_16'))
#
#         with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
#             init_fn(sess)
#             np_image, probabilities = sess.run([image, probabilities])
#             probabilities = probabilities[0, 0:]
#             sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
#
#         # plt.figure()
#         # plt.imshow(np_image.astype(np.uint8))
#         # plt.axis('off')
#         # plt.show()
#
#         names = imagenet.create_readable_names_for_imagenet_labels()
#         for i in range(5):
#             index = sorted_inds[i]
#             # Shift the index of a class name by one.
#             print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index + 1]))