import utils.preprocessing.inception_preprocessing as inception_preprocessing
import utils.preprocessing.vgg_preprocessing as vgg_preprocessing


def get_preprocessing(name, is_training=False):
    preprocessing_fn_map = {
        'inception_v1': inception_preprocessing,
        'inception_v4': inception_preprocessing,
        'inception_resnet_v2': inception_preprocessing,
        'mobilenet_v1': inception_preprocessing,
        'vgg': vgg_preprocessing,
        'vgg_a': vgg_preprocessing,
        'vgg_16': vgg_preprocessing,
        'vgg_19': vgg_preprocessing,
    }

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] is not known' % name)

    def preprocessing_fn(image, output_height, output_widht, **kwargs):
        return preprocessing_fn_map[name].preprocess_image(image, output_height, output_widht, is_training, **kwargs)

    return preprocessing_fn
