"""A function to build an object detection box coder from configuration."""
import model.detection.box_coders.faster_rcnn_box_coder as faster_rcnn_box_coder
import model.detection.box_coders.mean_stddev_box_coder as mean_stddev_box_coder
import model.detection.box_coders.square_box_coder as square_box_coder
from model.detection import protos as box_coder_pb2


def build(box_coder_config):
    """Builds a box coder object based on the box coder config.

    Args:
      box_coder_config: A box_coder.proto object containing the config for the
        desired box coder.

    Returns:
      BoxCoder based on the config.

    Raises:
      ValueError: On empty box coder proto.
    """
    if not isinstance(box_coder_config, box_coder_pb2.BoxCoder):
        raise ValueError('box_coder_config not of type box_coder_pb2.BoxCoder.')

    if box_coder_config.WhichOneof('box_coder_oneof') == 'faster_rcnn_box_coder':
        return faster_rcnn_box_coder.FasterRcnnBoxCoder(scale_factors=[
            box_coder_config.faster_rcnn_box_coder.y_scale,
            box_coder_config.faster_rcnn_box_coder.x_scale,
            box_coder_config.faster_rcnn_box_coder.height_scale,
            box_coder_config.faster_rcnn_box_coder.width_scale
        ])
    if (box_coder_config.WhichOneof('box_coder_oneof') ==
            'mean_stddev_box_coder'):
        return mean_stddev_box_coder.MeanStddevBoxCoder()
    if box_coder_config.WhichOneof('box_coder_oneof') == 'square_box_coder':
        return square_box_coder.SquareBoxCoder(scale_factors=[
            box_coder_config.square_box_coder.y_scale,
            box_coder_config.square_box_coder.x_scale,
            box_coder_config.square_box_coder.length_scale
        ])
    raise ValueError('Empty box coder.')
