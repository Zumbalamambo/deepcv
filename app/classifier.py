# import urllib2 as urllib
from PIL import Image
import tensorflow as tf
import model.classification.classifier_factory as classfier_factory
import utils.preprocessing.preprocessing_factory as preprocessing_factory

slim = tf.contrib.slim


def classify_image(config, args):
    # parse arguments
    model_name = config.get('config', 'model')
    # ds_name = config.get('dataset', 'name')
    num_class = config.get('dataset', 'num_class')
    img_height = config.get(model_name, 'height')
    img_width = config.get(model_name, 'width')
    ckpt_path = args.ckpt

    # load file
    image_ph = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
    image_path = args.file
    image = Image.open(image_path)
    # image_str = urllib.urlopen(image_path).read()
    # image = tf.image.decode_jpeg(image_str, channels=3)
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

    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_model_variables(model_name))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init_fn(sess)
        classify_result = sess.run([category_probablity],{image_ph: image})

    print(classify_result)