"""Interface for data decoders.

Data decoders decode the input data and return a dictionary of tensors keyed by
the entries in core.reader.Fields.
"""
from abc import ABCMeta
from abc import abstractmethod


class DataDecoder(object):
    """Interface for data decoders."""
    __metaclass__ = ABCMeta

    # TODO: snake_case this method.
    @abstractmethod
    def Decode(self, data):
        """Return a single image and associated labels.

        Args:
          data: a string tensor holding a serialized protocol buffer corresponding
            to data for a single image.

        Returns:
          tensor_dict: a dictionary containing tensors. Possible keys are defined in
              reader.Fields.
        """
        pass
