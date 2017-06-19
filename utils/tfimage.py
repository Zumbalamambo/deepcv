import inspect
import numpy as np
from PIL import Image
import tensorflow as tf


def verify_imageshape(imagepath, imageshape):
    with Image.open(imagepath) as image:
        return np.all(np.equal(image.size, imageshape[1::-1]))


def verify_image_jpeg(image_path, image_shape):
    scope = inspect.stack()[0][3]
    try:
        graph = tf.get_default_graph()
        path = graph.get_tensor_by_name(scope + '/path:0')
        decode = graph.get_tensor_by_name(scope + '/decode_jpeg:0')
    except KeyError:
        tf.logging.debug('creating decode_jpeg tensor')
        path = tf.placeholder(tf.string, name=scope + '/path')
        image_file = tf.read_file(path, name=scope + '/read_file')
        decode = tf.image.decode_jpeg(image_file, channels=3, name=scope + '/decode_jpeg')

    try:
        image = tf.get_default_session().run(decode, {path: image_path})
    except:
        return False
    return np.all(np.equal(image.shape[:2], image_shape[:2]))


def check_coords(objects_coord):
    return np.all(objects_coord[:, 0] <= objects_coord[:, 2]) and np.all(objects_coord[:, 1] <= objects_coord[:, 3])


def verify_coords(objects_coord, imageshape):
    assert check_coords(objects_coord)
    return np.all(objects_coord >= 0) and np.all(objects_coord <= np.tile(imageshape[1::-1], [2]))


def fix_coords(objects_coord, imageshape):
    assert check_coords(objects_coord)
    objects_coord = np.maximum(objects_coord, np.zeros([4], dtype=objects_coord.dtype))
    objects_coord = np.minimum(objects_coord, np.tile(np.asanyarray(imageshape[1::-1], objects_coord.dtype), [2]))
    return objects_coord


def per_image_standardization(image):
    stddev = np.std(image)
    return (image - np.mean(image)) / max(stddev, 1.0 / np.sqrt(np.multiply.reduce(image.shape)))


def flip_horizontally(image, objects_coord, width):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        image = tf.image.flip_left_right(image)
        xmin, ymin, xmax, ymax = objects_coord[:, 0:1], objects_coord[:, 1:2], \
                                 objects_coord[:, 2:3], objects_coord[:, 3:4]
        objects_coord = tf.concat([width - xmax, ymin, width - xmin, ymax], 1)

    return image, objects_coord


def random_crop(image, objects_coord, width_height, scale=1):
    assert 0 < scale <= 1
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
         xy_min = tf.reduce_min(objects_coord[:, :2], 0)
         xy_max = tf.reduce_max(objects_coord[:, 2:], 0)
         margin = width_height - xy_max
         shrink = tf.random_uniform([4], maxval=scale) * tf.concat([xy_min, margin], 0)
         _xy_min = shrink[:2]
         _wh = width_height - shrink[2:] -_xy_min
         objects_coord = objects_coord - tf.tile(_xy_min, [2])
         _xy_min_ = tf.cast(_xy_min, tf.int32)
         _wh_ = tf.cast(_wh, tf.int32)
         image = tf.image.crop_to_bounding_box(image, _xy_min_[1], _xy_min_[0], _wh_[1], _wh_[0])

    return image, objects_coord, _wh


def random_flip_horizontally(image, objects_coord, width, probability=0.5):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: flip_horizontally(image, objects_coord, width)
        fn2 = lambda: (image, objects_coord)

    return tf.cond(pred, fn1, fn2)


def random_grayscale(image, probability=0.5):
    if probability < 0:
        return image
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: tf.tile(tf.image.rgb_to_grayscale(image), [1] * (len(image.get_shape()) - 1) + [3])
        fn2 = lambda: image
    return tf.cond(pred, fn1, fn2)