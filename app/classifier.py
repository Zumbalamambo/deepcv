import os
from PIL import Image
import json
import numpy as np
import tensorflow as tf
import model.classification.classifier_factory as classfier_factory
import utils.preprocessing.preprocessing_factory as preprocessing_factory
import utils.dataset.imagenet as imagenet
import model.classification.inception_v1 as inception_v1
try:
    import urllib2
except ImportError:
    import urllib.request as urllib

slim = tf.contrib.slim


def run(config, args):
    if args.file is not None:
        classify_image_local(config, args)
    elif args.file_url is not None:
        result = classify_iamge_remote(config, args)
        print(result)
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

    # load file
    image_ph = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
    image_original = Image.open(image_path)
    image = image_original.resize((img_height, img_width))
    image = np.expand_dims(np.array(np.uint8(image)), 0)

    feed_dict = {image_ph: image}

    net_fn = classfier_factory.get_network_fn(model_name, num_class, is_training=False)
    # logits, _ = inception_v1.inception_v1(image_ph, num_class, is_training=False)
    logits, _ = net_fn(image_ph)
    net_out = tf.nn.softmax(logits)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        if os.path.isdir(ckpt_path):
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_model_variables(model_variables))
            init_fn(sess)

        classify_result = sess.run(net_out, feed_dict=feed_dict)

        probablities = classify_result[0, 0:]
        sorted_id = [i[0] for i in sorted(enumerate(-probablities), key=lambda x: x[1])]

    if args.json:
        top5_result = []
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_id[i]
        print('Probablity %.2f%% ==> [%s]' % ((probablities[index] * 100), names[index + 1]))
        if args.json:
            top5_result.append({'name': names[index+1], 'probability': probablities[index] * 100})

    if args.json:
        return json.dumps(top5_result)


def classify_iamge_remote(config, args):
    # parse arguments
    model_name = config.get('config', 'model')
    # ds_name = config.get('dataset', 'name')
    num_class = int(config.get('dataset', 'num_class'))
    img_height = int(config.get(model_name, 'height'))
    img_width = int(config.get(model_name, 'width'))
    ckpt_path = config.get('weights', 'ckpt')
    image_url = args.file_url

    image_str = urllib.urlopen(image_url).read()
    image = tf.image.decode_jpeg(image_str, channels=3)

    preprocess_image_fn = preprocessing_factory.get_preprocessing(model_name, is_training=True)

    processed_image = preprocess_image_fn(image, img_height, img_width)
    processed_image = tf.expand_dims(processed_image, 0)

    net_fn = classfier_factory.get_network_fn(model_name, num_class, is_training=False)
    logits, _ = net_fn(processed_image)
    net_out = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_model_variables(model_name))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init_fn(sess)
        np_image, probablities = sess.run([image, net_out])
        probablities = probablities[0, 0:]
        sorted_id = [i[0] for i in sorted(enumerate(-probablities), key=lambda x: x[1])]

    top5_result = []
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_id[i]
        top5_result.append({'name': names[index + 1], 'probability': probablities[index] * 100})

    return json.dumps(top5_result)

