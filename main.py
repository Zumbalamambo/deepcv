import argparse
import configparser

import tensorflow as tf

import app.yolo as yolo
import app.classifier as classifier
import utils.tfsys as tfsys


def main():
    parser = argparse.ArgumentParser("[DeepCV]")
    # run app with base arguments
    parser.add_argument('--config', default='', help='config file')
    parser.add_argument('--app', type=str, default='mnist', help='')
    parser.add_argument('--task', type=str, default='predict', help='')
    parser.add_argument('--file', help='local image path')
    parser.add_argument('--file_url', help='remote image path')

    # result type
    parser.add_argument('--json', default=False, help='')
    parser.add_argument('--visible', default=False, help='')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    tfsys.load_config(config, args.config)

    if args.level:
        tf.logging.set_verbosity(args.level.upper())

    if args.app == 'tfrecord':
        import utils.tfrecord as tfrecord
        if args.datatype == 'voc':
            tfrecord.voc_to_tfrecord(config, args)
        else:
            print('%s cannot convert into tfrecord' % args.datatype)
    elif args.app == 'classifier':
        classifier.run(config, args)
    elif args.app == 'detector':
        import app.detector as detector
        detector.run(config, args)
    elif args.app == 'yolo':
        yolo.run(config, args)
    else:
        print("No this app!")


if __name__ == '__main__':
    main()
