import utils.dataset.voc as voc


def voc_to_tfrecord(config, args):
    voc.convert_to_tfrecord(config, args)