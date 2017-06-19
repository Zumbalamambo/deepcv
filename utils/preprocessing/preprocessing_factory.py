import utils.preprocessing.inception_preprocessing as inception_preprocessing


def get_preprocessing(name, is_training):
    preprocessing_fn_map = {
        'mobilenet_v1': inception_preprocessing
    }

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] is not known' % name)

    def preprocessing_fn(image, output_height, output_widht, **kwargs):
        return preprocessing_fn_map[name].preprocess_image(image, output_height, output_widht, is_training, **kwargs)

    return preprocessing_fn
