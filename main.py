import argparse
import configparser
import time
import app.yolo as yolo
import app.classifier as classifier
import utils.tfsys as tfsys


def main():
    parser = argparse.ArgumentParser("[DeepCV]")
    # run app with base arguments
    parser.add_argument('--config', default='', help='config file')
    parser.add_argument('--app', type=str, default='', help='')
    parser.add_argument('--task', type=str, default='', help='')
    parser.add_argument('--file', default=None, help='local image path')
    # eg. --file_url=https://upload.wikimedia.org/wikipedia/commons/d/d9/First_Student_IC_school_bus_202076.jpg
    parser.add_argument('--file_url', default=None, help='remote image path')
    parser.add_argument('--gpu', type=bool, default=False, help='')
    # result type
    parser.add_argument('--print', type=bool, default=True, help='')
    parser.add_argument('--show', type=bool, default=False, help='')
    parser.add_argument('--json', type=bool, default=False, help='')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    tfsys.load_config(config, args.config)
    start_time = time.time()
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

    print("App cost time %.2f s" % (time.time()-start_time))

if __name__ == '__main__':
    main()
