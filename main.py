import time
import argparse
import configparser
import tensorflow as tf
import utils.tfdata as tfdata
import utils.tfsys as tfsys
import app.yolo as yolo
import app.classifier as classifier

def main():
    parser = argparse.ArgumentParser("[DeepCV]")
    parser.add_argument('--file', help='input image path')
    parser.add_argument('--app', type=str, default='mnist', help='')
    parser.add_argument('--task', type=str, default='classify', help='')
    parser.add_argument('--model', type=str, default='vgg16', help='')
    parser.add_argument('--gpu', type=bool, default=False, help='')
    parser.add_argument('-c', '--config', nargs='+', default=['./config.cfg'], help='config file')
    parser.add_argument('--ckpt', help='')
    parser.add_argument('-l', '--logdir', help='loading model from a .ckpt file')
    parser.add_argument('-d', '--delete', action='store_true', help='delete logdir')

    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val'], help='')

    # cache data
    parser.add_argument('-v', '--verify', default=True, action='store_true')
    parser.add_argument('-ds', '--dataset_name', default='cifar10', help='')
    # training params
    parser.add_argument('-tf', '--transfer', help='transferring model from a .ckpt file')
    parser.add_argument('-ec', '--exclude', nargs='+', help='exclude variables while transferring')

    parser.add_argument('-b', '--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('-o', '--optimizer', default='adam')
    parser.add_argument('-lr', '--learning_rate', default=1e-6, type=float, help='learning rate')
    parser.add_argument('-s', '--steps', type=int, default=None, help='max number of steps')
    parser.add_argument('--summary_secs', default=30, type=int, help='seconds to save summaries')
    parser.add_argument('--save_secs', default=60, type=int, help='seconds to save model')

    # detection configure
    parser.add_argument('-th', '--threshold', type=float, default=0.3)
    parser.add_argument('-thi', '--threshold_iou', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('--level', default='info', help='logging level')
    parser.add_argument('-n', '--logname', default=time.strftime('%Y-%m-%d_%H-%M-%S'), help='the name for TensorBoard')
    parser.add_argument('-g', '--gradient_clip', default=0, type=float, help='gradient clip')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--master', default='', help='master address')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    tfsys.load_config(config, args.config)

    if args.level:
        tf.logging.set_verbosity(args.level.upper())

    if args.app == 'data':
        tfdata.download_covert2record(args)
    elif args.app == 'classify':
        classifier.run(config, args)
    elif args.app == 'mnist':
        # mnist.run(args)
        pass
    elif args.app == 'cifar':
        pass
    elif args.app == 'yolo':
        yolo.run(config, args)
    else:
        print("No this app!")


if __name__ == '__main__':
    main()
