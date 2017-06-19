import os
import sys
import tqdm
import bs4
import numpy as np
import tensorflow as tf
import utils.tfimage as tfimage


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
