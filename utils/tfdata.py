import utils.dataset.cifar10 as cifar10
import utils.dataset.imagenet as imagenet
import utils.dataset.voc as voc
import utils.dataset.pet as pet


datasets_map = {'cifar10': cifar10,
                'imagenet': imagenet,
                'voc': voc,
                'pet': pet
                }


def download_convert(config, args):
    if args.dataset_name not in datasets_map:
        raise ValueError('%s dataset is unkown' % args.name)
    datasets_map[args.dataset_name].convert_to_tfrecord(config)


# def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
#     if name not in datasets_map:
#         raise ValueError('%s dataset is unknown' % name)
#     return datasets_map[name].get_split(split_name, dataset_dir, file_pattern, reader)