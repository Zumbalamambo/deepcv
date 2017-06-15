import os
import pickle
import gzip
import importlib
import numpy as np
import pandas as pandas
import tensorflow as tf
import utils.tfsys as tfsys


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        # detection = pickle.load(fp,encoding='iso-8859-1')
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data


def exists_or_mkdir(path, verbose=True):
    if not os.path.exists(path):
        if verbose:
            print("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            print("[!] %s exists ..." % path)
        return True


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets), 'the length of X,y is not same'
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


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

    exists_or_mkdir(working_directory, verbose=False)
    filepath = os.path.join(working_directory, filename)

    if not os.path.exists(filepath):
        _download(filename, working_directory, url_source)
        print()
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


def load_mnist_dataset(shape=(-1, 784), path="/home/sunxl/dataset/mnist"):
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


def cache(config, args):
    label_file = config.get('dataset', 'label')
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f]

    labels_index = dict([(name, i) for i, name in enumerate(labels)])
    dataset = [(os.path.basename(os.path.splitext(path)[0]), pandas.read_csv(os.path.expanduser(os.path.expandvars(path)))) \
               for path in config.get('dataset', 'data').split(':')]

    module = importlib.import_module('utils.data.cache')
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

