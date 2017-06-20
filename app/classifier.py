from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model.classification.classifier_factory as classfier_factory
import utils.preprocessing.preprocessing_factory as preprocessing_factory
import utils.dataset.imagenet as imagenet

try:
    import urllib2
except ImportError:
    import urllib.request as urllib

slim = tf.contrib.slim


def run(config, args):
    classify_image_local(config, args)


def classify_image_local(config, args):
    # parse arguments
    model_name = config.get('config', 'model')
    # ds_name = config.get('dataset', 'name')
    num_class = int(config.get('dataset', 'num_class'))
    img_height = int(config.get(model_name, 'height'))
    img_width = int(config.get(model_name, 'width'))
    ckpt_path = args.ckpt

    # load file
    image_ph = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
    image_path = args.file
    image_original = Image.open(image_path)
    image = image_original.resize((img_height, img_width))
    image = np.expand_dims(np.array(np.uint8(image)), 0)

    net = classfier_factory.get_network_fn(model_name, num_class, is_training=False)

    logits, _ = net(image_ph)
    net_out = tf.nn.softmax(logits)
    print(model_name)
    # model_path = tf.train.latest_checkpoint(ckpt_path)
    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_model_variables(model_name))
    # init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_variables_to_restore())
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init_fn(sess)
        classify_result = sess.run(net_out, {image_ph: image})
        probablities = classify_result[0, 0:]

        sorted_id = [i[0] for i in sorted(enumerate(-probablities), key=lambda x: x[1])]

    top5_names = []
    top5_probablities = []
    top5_result = []
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_id[i]
        top5_names.append(str(names[index]))
        top5_probablities.append(probablities[index] * 100)
        top5_result.append('Probablity %.2f%% ==> [%s] \r' % ((probablities[index] * 100), names[index]))
        print('Probablity %.2f%% ==> [%s]' % ((probablities[index] * 100), names[index]))


def classify_iamge_remote(config, args):
    # parse arguments
    model_name = config.get('config', 'model')
    # ds_name = config.get('dataset', 'name')
    num_class = int(config.get('dataset', 'num_class'))
    img_height = int(config.get(model_name, 'height'))
    img_width = int(config.get(model_name, 'width'))
    ckpt_path = args.ckpt
    image_url = args.image_url

    image_str = urllib.urlopen(image_url).read()
    image = tf.image.decode_jpeg(image_str, channels=3)
    process_image = preprocessing_factory.get_preprocessing(model_name)
    image = process_image(image, img_height, img_width, is_training=True)
    image = np.expand_dims(np.array(np.uint8(image)), 0)

    net = classfier_factory.get_network_fn(model_name, num_class, is_training=False)
    logits, _ = net(image)
    net_out = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_model_variables(model_name))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        np_image, probablities = sess.run([image, net_out])
        probablities = probablities[0, 0:]
        sorted_id = [i[0] for i in sorted(enumerate(probablities), key=lambda x: x[0])]

