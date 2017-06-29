import utils.dataset.cifar10 as cifar10
import utils.dataset.imagenet as imagenet


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    datasets_map = {
        'cifar10': cifar10,
        'imagenet': imagenet,
    }
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)

    return datasets_map[name].get_split(split_name, dataset_dir, file_pattern, reader)
