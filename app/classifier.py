# import urllib2 as urllib
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model.classification.classifier_factory as classfier_factory
import utils.preprocessing.preprocessing_factory as preprocessing_factory
import utils.dataset.imagenet as imagenet

slim = tf.contrib.slim


def classify_image(config, args):
    # parse arguments
    model_name = config.get('config', 'model')
    # ds_name = config.get('dataset', 'name')
    num_class = config.get('dataset', 'num_class')
    img_height = int(config.get(model_name, 'height'))
    img_width = int(config.get(model_name, 'width'))
    ckpt_path = args.ckpt

    # load file
    image_ph = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
    image_path = args.file
    image_original = Image.open(image_path)
    image = image_original.resize((img_height, img_width))
    image = np.expand_dims(np.array(np.uint8(image)), 0)
    # print(image.shape)
    # image_str = urllib.urlopen(image_path).read()
    # image = tf.image.decode_jpeg(image, channels=3)
    # preprocessing = preprocessing_factory.get_preprocessing(model_name, is_training=False)
    # processed_image = preprocessing(image, img_height, img_width)
    # processed_image = tf.expand_dims(processed_image, 0)

    ####################
    # Select the model #
    ####################
    net = classfier_factory.get_network_fn(model_name, num_class, is_training=False)

    # with slim.arg_scope(classfier_factory.arg_scopes_map[model_name]):
    logits, _ = net(image_ph)
    category_probablity = tf.nn.softmax(logits)
    print(ckpt_path)
    # model_path = tf.train.latest_checkpoint(ckpt_path)
    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_variables_to_restore())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init_fn(sess)
        classify_result = sess.run(category_probablity, {image_ph: image})
        probablities = classify_result[0, 0:]

        sorted_id = [i[0] for i in sorted(enumerate(-probablities), key=lambda x: x[1])]

    names = imagenet.create_readable_names_for_imagenet_labels()

    for i in range(5):
        index = sorted_id[i]
        print('Probablity %.10f%% ==> [%s]' % ((probablities[index] * 100), names[index]))

    benchmark = max(probablities)
    for k, v in enumerate(probablities):
        if v == benchmark:
            print(names[k])
        else:
            continue

    plt.figure()
    plt.imshow(image_original)
    plt.axis('off')
    plt.show()

