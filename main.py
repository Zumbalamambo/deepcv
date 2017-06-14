import argparse
import configparser
import tensorflow as tf
import utils.tfdata as tfdata
import utils.tfsys as tfsys
import app.mnist as mnist
import app.demo_yolo as demo_yolo


def main():
    parser = argparse.ArgumentParser("[DeepCV]")
    parser.add_argument('--file', help='input image path')
    parser.add_argument('--app', type=str, default='mnist', help='')
    parser.add_argument('--task', type=str, default='classify', help='')
    parser.add_argument('--model', type=str, default='vgg16', help='')
    parser.add_argument('--gpu', type=bool, default=False, help='')

    # detection configure
    parser.add_argument('-c', '--config', nargs='+', default=['./config/config.ini'], help='config file')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val', 'test'], help='')
    parser.add_argument('-v', '--verify', action='store_true')
    parser.add_argument('-t', '--threshold', type=float, default=0.3)
    parser.add_argument('--threshold_iou', type=float, default=0.4, help='IoU threshold')

    parser.add_argument('--level', default='info', help='logging level')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    tfsys.load_config(config, args.config)

    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    if args.app == 'data':
        tfdata.cache(config, args)
    elif args.app == 'mnist':
        mnist.run(args)
    elif args.app == 'demo_yolo':
        demo_yolo.run(config, args)
    else:
        print("No this app!")


if __name__ == '__main__':
    main()
