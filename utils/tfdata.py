import os
import sys
import tqdm
import gzip
import tarfile
import zipfile
import bs4
import importlib
import inspect
import numpy as np
import pandas as pandas
from PIL import Image
import tensorflow as tf
import utils.tfsys as tfsys
import utils.tfimage as tfimage


def maybe_download_and_extract(filename, working_directory, url_source, extract=False, expected_bytes=None):
    # We first define a download function, supporting both Python 2 and 3.
    def _download(filename, working_directory, url_source):
        def _dlProgress(count, blockSize, totalSize):
            if (totalSize != 0):
                percent = float(count * blockSize) / float(totalSize) * 100.0
                sys.stdout.write("\r" "Downloading " + filename + "...%d%%" % percent)
                sys.stdout.flush()

        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve
        filepath = os.path.join(working_directory, filename)
        urlretrieve(url_source + filename, filepath, reporthook=_dlProgress)

    tfsys.exists_or_mkdir(working_directory, verbose=False)
    filepath = os.path.join(working_directory, filename)

    if not os.path.exists(filepath):
        _download(filename, working_directory, url_source)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if (not (expected_bytes is None) and (expected_bytes != statinfo.st_size)):
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
        if (extract):
            if tarfile.is_tarfile(filepath):
                print('Trying to extract tar file')
                tarfile.open(filepath, 'r').extractall(working_directory)
                print('... Success!')
            elif zipfile.is_zipfile(filepath):
                print('Trying to extract zip file')
                with zipfile.ZipFile(filepath) as zf:
                    zf.extractall(working_directory)
                print('... Success!')
            else:
                print("Unknown compression_format only .tar.gz/.tar.bz2/.tar and .zip supported")
    return filepath


def load_mnist_dataset(shape=(-1, 784), path="~/dataset/mnist"):
    # We first define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    def load_mnist_images(path, filename):
        filepath = maybe_download_and_extract(filename, path, 'http://yann.lecun.com/exdb/mnist/')
        print(filepath)

        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(shape)

        return data / np.float32(256)

    def load_mnist_labels(path, filename):
        filepath = maybe_download_and_extract(filename, path, 'http://yann.lecun.com/exdb/mnist/')
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # Download and read the training and test set images and labels.
    print("Load or Download MNIST > {}".format(path))
    X_train = load_mnist_images(path, 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(path, 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(path, 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(path, 't10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def data_augmentation_full(image, objects_coord, width_height, config):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        random_crop = config.getfloat(section, 'random_crop')
        if random_crop > 0:
            image, objects_coord, width_height = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tfimage.random_crop(image, objects_coord, width_height, random_crop),
                lambda: (image, objects_coord, width_height)
            )
    return image, objects_coord, width_height


def resize_image_objects(image, objects_coord, width_height, width, height):
    with tf.name_scope(inspect.stack()[0][3]):
        image = tf.image.resize_images(image, [height, width])
        factor = [width, height] / width_height
        objects_coord = objects_coord * tf.tile(factor, [2])
    return image, objects_coord


def data_augmentation_resized(image, objects_coord, width, height, config):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        if config.getboolean(section, 'random_flip_horizontally'):
            image, objects_coord = tfimage.random_flip_horizontally(image, objects_coord, width)
        if config.getboolean(section, 'random_brightness'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_brightness(image, max_delta=63),
                lambda: image
            )
        if config.getboolean(section, 'random_saturation'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_saturation(image, lower=0.5, upper=1.5),
                lambda: image
            )
        if config.getboolean(section, 'random_hue'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_hue(image, max_delta=0.032),
                lambda: image
            )
        if config.getboolean(section, 'random_contrast'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_contrast(image, lower=0.5, upper=1.5),
                lambda: image
            )
        if config.getboolean(section, 'noise'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: image + tf.truncated_normal(tf.shape(image)) * tf.random_uniform([], 5, 15),
                lambda: image
            )
        grayscale_probability = config.getfloat(section, 'grayscale_probability')
        if grayscale_probability > 0:
            image = tfimage.random_grayscale(image, grayscale_probability)
    return image, objects_coord


def cache(config, args):
    label_file = config.get('dataset', 'label')
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f]

    labels_index = dict([(name, i) for i, name in enumerate(labels)])
    dataset = [(os.path.basename(os.path.splitext(path)[0]), pandas.read_csv(os.path.expanduser(os.path.expandvars(path)))) \
               for path in config.get('dataset', 'data').split(':')]

    module = importlib.import_module('utils.tfdata')
    cache_dir = tfsys.get_cachedir(config)
    data_dir = os.path.join(cache_dir, 'dataset', config.get('dataset', 'name'))
    os.makedirs(data_dir, exist_ok=True)

    for profile in args.profile:
        tfrecord_file = os.path.join(data_dir, profile+'.tfrecord')
        tf.logging.info('Write tfrecord file:' + tfrecord_file)
        with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
            for name, data in dataset:
                func = getattr(module, name)
                for i, row in data.iterrows():
                    print(row)
                    func(writer, labels_index, profile, row, args.verify)


def coco(writer, labels_index, profile, row, verify=True):
    # load annotation file
    root = os.path.expanduser(os.path.expandvars(row['root']))
    name = profile + '2014'
    data_dir = os.path.join(root, name)
    annotation_path = os.path.join(root, 'annotations', 'instances_%s.json' % name)
    print('annotation path: ' + annotation_path)
    if not os.path.exists(annotation_path):
        tf.logging.warn(annotation_path + ' not exists')
        return False

    import pycocotools.coco
    coco = pycocotools.coco.COCO(annotation_path)
    cat_ids = coco.getCatIds(catNms=list(labels_index.keys()))
    cats = coco.loadCats(cat_ids)
    id_index = dict((cat['id'], labels_index[cat['name']]) for cat in cats)
    img_ids = coco.getImgIds()

    imgs = coco.loadImgs(img_ids)
    _imgs = list(filter(lambda img: os.path.exists(os.path.join(data_dir, img['file_name'])), imgs))

    if len(imgs) > len(_imgs):
        tf.logging.warn('%d of %d images not exists' % (len(imgs) - len(_imgs), len(imgs)))

    cnt_noobj = 0
    decode_error = 0
    for img in tqdm.tqdm(_imgs):
        ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if len(anns) <= 0:
            cnt_noobj += 1
            continue
        image_path = os.path.join(data_dir, img['file_name'])
        # print(image_path)
        width, height = img['width'], img['height']
        imageshape = [height, width, 3]
        objects_class = np.array([id_index[ann['category_id']] for ann in anns], dtype=np.int64)
        objects_coord = [ann['bbox'] for ann in anns]
        objects_coord = [(x, y, x + w, y + h) for x, y, w, h in objects_coord]
        objects_coord = np.array(objects_coord, dtype=np.float32)

        if False:
            if not tfimage.verify_coords(objects_coord, imageshape):
                # tf.logging.error('failed to verify coordinates of ' + imagepath)
                continue
            if not tfimage.verify_image_jpeg(image_path, imageshape):
                # tf.logging.error('failed to decode ' + imagepath)
                decode_error += 1
                continue

        assert len(objects_class) == len(objects_coord)

        example = tf.train.Example(features=tf.train.Features(feature={
            'imagepath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(image_path)])),
            'imageshape': tf.train.Feature(int64_list=tf.train.Int64List(value=imageshape)),
            'objects': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[objects_class.tostring(), objects_coord.tostring()])),
        }))
        writer.write(example.SerializeToString())

    if cnt_noobj > 0:
        tf.logging.warn('%d of %d images have no object' % (cnt_noobj, len(_imgs)))

    if decode_error > 0:
        tf.logging.warn('%d of %d images decode error' % (decode_error, len(_imgs)))

    return True


def load_voc_annotation(path, name_index):
    with open(path, 'r') as f:
        anno = bs4.BeautifulSoup(f.read(), 'xml').find('annotation')
    objects_class = []
    objects_coord = []
    for obj in anno.find_all('object', recursive=False):
        for bndbox, name in zip(obj.find_all('bndbox', recursive=False), obj.find_all('name', recursive=False)):
            if name.text in name_index:
                objects_class.append(name_index[name.text])
                xmin = float(bndbox.find('xmin').text) - 1
                ymin = float(bndbox.find('ymin').text) - 1
                xmax = float(bndbox.find('xmax').text) - 1
                ymax = float(bndbox.find('ymax').text) - 1
                objects_coord.append((xmin, ymin, xmax, ymax))
            else:
                sys.stderr.write(name.text + ' not in names')
    size = anno.find('size')
    return anno.find('filename').text, \
           (int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)), \
           objects_class, \
           objects_coord


def voc(writer, name_index, profile, row, verify=True):
    root = os.path.expanduser(os.path.expandvars(row['root']))
    path = os.path.join(root, 'ImageSets', 'Main', profile) + '.txt'
    if not os.path.exists(path):
        tf.logging.warn(path + ' not exists')
        return False
    with open(path, 'r') as f:
        filenames = [line.strip() for line in f]
    annotations = [os.path.join(root, 'Annotations', filename + '.xml') for filename in filenames]
    _annotations = list(filter(os.path.exists, annotations))
    if len(annotations) > len(_annotations):
        tf.logging.warn('%d of %d images not exists' % (len(annotations) - len(_annotations), len(annotations)))
    cnt_noobj = 0
    for path in tqdm.tqdm(_annotations):
        imagename, imageshape, objects_class, objects_coord = load_voc_annotation(path, name_index)
        if len(objects_class) <= 0:
            cnt_noobj += 1
            continue
        objects_class = np.array(objects_class, dtype=np.int64)
        objects_coord = np.array(objects_coord, dtype=np.float32)
        imagepath = os.path.join(root, 'JPEGImages', imagename)
        if verify:
            if not tfimage.verify_coords(objects_coord, imageshape):
                tf.logging.error('failed to verify coordinates of ' + imagepath)
                continue
            if not tfimage.verify_image_jpeg(imagepath, imageshape):
                tf.logging.error('failed to decode ' + imagepath)
                continue
        assert len(objects_class) == len(objects_coord)
        example = tf.train.Example(features=tf.train.Features(feature={
            'imagepath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(imagepath)])),
            'imageshape': tf.train.Feature(int64_list=tf.train.Int64List(value=imageshape)),
            'objects': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[objects_class.tostring(), objects_coord.tostring()])),
        }))
        # print(example['ima'])
        writer.write(example.SerializeToString())
    if cnt_noobj > 0:
        tf.logging.warn('%d of %d images have no object' % (cnt_noobj, len(filenames)))
    return True


def transform_labels(objects_class, objects_coord, classes, cell_width, cell_height, dtype=np.float32):
    cells = cell_height * cell_width
    mask = np.zeros([cells, 1], dtype=dtype)
    prob = np.zeros([cells, 1, classes], dtype=dtype)
    coords = np.zeros([cells, 1, 4], dtype=dtype)
    offset_xy_min = np.zeros([cells, 1, 2], dtype=dtype)
    offset_xy_max = np.zeros([cells, 1, 2], dtype=dtype)
    assert len(objects_class) == len(objects_coord)
    xmin, ymin, xmax, ymax = objects_coord.T
    x = cell_width * (xmin + xmax) / 2
    y = cell_height * (ymin + ymax) / 2
    ix = np.floor(x)
    iy = np.floor(y)
    offset_x = x - ix
    offset_y = y - iy
    w = xmax - xmin
    h = ymax - ymin
    index = (iy * cell_width + ix).astype(np.int)
    mask[index, :] = 1
    prob[index, :, objects_class] = 1
    coords[index, 0, 0] = offset_x
    coords[index, 0, 1] = offset_y
    coords[index, 0, 2] = np.sqrt(w)
    coords[index, 0, 3] = np.sqrt(h)
    _w = w / 2 * cell_width
    _h = h / 2 * cell_height
    offset_xy_min[index, 0, 0] = offset_x - _w
    offset_xy_min[index, 0, 1] = offset_y - _h
    offset_xy_max[index, 0, 0] = offset_x + _w
    offset_xy_max[index, 0, 1] = offset_y + _h
    wh = offset_xy_max - offset_xy_min
    assert np.all(wh >= 0)
    areas = np.multiply.reduce(wh, -1)
    return mask, prob, coords, offset_xy_min, offset_xy_max, areas


def decode_labels(objects_class, objects_coord, classes, cell_width, cell_height):
    with tf.name_scope(inspect.stack()[0][3]):
        mask, prob, coords, offset_xy_min, offset_xy_max, areas = tf.py_func(transform_labels,
                                                                             [objects_class, objects_coord, classes,
                                                                              cell_width, cell_height],
                                                                             [tf.float32] * 6)
        cells = cell_height * cell_width
        with tf.name_scope('reshape_labels'):
            mask = tf.reshape(mask, [cells, 1], name='mask')
            prob = tf.reshape(prob, [cells, 1, classes], name='prob')
            coords = tf.reshape(coords, [cells, 1, 4], name='coords')
            offset_xy_min = tf.reshape(offset_xy_min, [cells, 1, 2], name='offset_xy_min')
            offset_xy_max = tf.reshape(offset_xy_max, [cells, 1, 2], name='offset_xy_max')
            areas = tf.reshape(areas, [cells, 1], name='areas')

    return mask, prob, coords, offset_xy_min, offset_xy_max, areas


def decode_images(paths):
    with tf.name_scope(inspect.stack()[0][3]):
        with tf.name_scope('parse_example'):
            reader = tf.TFRecordReader()
            _, serialized = reader.read(tf.train.string_input_producer(paths))
            example = tf.parse_single_example(serialized, features={
                'imagepath': tf.FixedLenFeature([], tf.string),
                'imageshape': tf.FixedLenFeature([3], tf.int64),
                'objects': tf.FixedLenFeature([2], tf.string),
            })

        image_path = example['imagepath']
        with tf.name_scope('load_image'):
            image_file = tf.read_file(image_path)
            image = tf.image.decode_jpeg(image_file, channels=3)

        objects = example['objects']
        with tf.name_scope('decode_objects'):
            objects_class = tf.decode_raw(objects[0], tf.int64, name='objects_class')
            objects_coord = tf.decode_raw(objects[1], tf.float32)
            objects_coord = tf.reshape(objects_coord, [-1, 4], name='objects_coord')

    return image, example['imageshape'], objects_class, objects_coord


def load_image_labels(paths, classes, width, height, cell_width, cell_height, config):
    with tf.name_scope('batch'):
        # image
        image, imageshape, objects_class, objects_coord = decode_images(paths)
        image = tf.cast(image, tf.float32)
        width_height = tf.cast(imageshape[1::-1], tf.float32)
        if config.getboolean('data_augmentation_full', 'enable'):
            image, objects_coord, width_height = data_augmentation_full(image, objects_coord, width_height, config)
        image, objects_coord = resize_image_objects(image, objects_coord, width_height, width, height)
        if config.getboolean('data_augmentation_resized', 'enable'):
            image, objects_coord = data_augmentation_resized(image, objects_coord, width, height, config)
        image = tf.clip_by_value(image, 0, 255)
        objects_coord = objects_coord / [width, height, width, height]

        # labels
        labels = decode_labels(objects_class, objects_coord, classes, cell_width, cell_height)

    return image, labels