import os
import sys
import tqdm
import bs4
import hashlib
import io
import logging
from lxml import etree
import PIL.Image
import numpy as np
import tensorflow as tf
import utils.tfimage as tfimage
import utils.detection.dataset_util as dataset_util
import utils.detection.label_map_util as label_map_util


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


def dict_to_tf_example(data, dataset_directory, label_map_dict, ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)

    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def convert_to_tfrecord(config, args):
    year = args.year
    set = args.set

    data_dir = config.get('data', 'directory')
    if set == 'train' or set == 'val':
        sub_set = 'train_val'
    elif set == 'test':
        sub_set = 'test'
    else:
        print('No this dir!')
    data_dir = os.path.join(data_dir, sub_set)

    label_map_path = config.get('label', 'path')
    tfrecord_dir = config.get('tfrecord', 'directory')
    tfrecord_path = os.path.join(tfrecord_dir, "%s_%s.tfrecords" % (set, year))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    logging.info('Reading from PASCAL %s dataset.', year)

    examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main', 'aeroplane_' + set + '.txt')
    annotations_dir = os.path.join(data_dir, year, 'Annotations')
    examples_list = dataset_util.read_examples_list(examples_path)

    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(annotations_dir, example + '.xml')
        # print(path)
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        # print(xml_str)
        xml = etree.fromstring(xml_str)
        # print(xml)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        # print(data)
        tf_example = dict_to_tf_example(data, data_dir, label_map_dict, False)
        writer.write(tf_example.SerializeToString())

    writer.close()
