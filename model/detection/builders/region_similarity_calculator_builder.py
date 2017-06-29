"""Builder for region similarity calculators."""

import model.detection.core.region_similarity_calculator as region_similarity_calculator
from model.detection import protos as region_similarity_calculator_pb2


def build(region_similarity_calculator_config):
    """Builds region similarity calculator based on the configuration.

    Builds one of [IouSimilarity, IoaSimilarity, NegSqDistSimilarity] objects. See
    core/region_similarity_calculator.proto for details.

    Args:
      region_similarity_calculator_config: RegionSimilarityCalculator
        configuration proto.

    Returns:
      region_similarity_calculator: RegionSimilarityCalculator object.

    Raises:
      ValueError: On unknown region similarity calculator.
    """

    if not isinstance(
            region_similarity_calculator_config,
            region_similarity_calculator_pb2.RegionSimilarityCalculator):
        raise ValueError(
            'region_similarity_calculator_config not of type '
            'region_similarity_calculator_pb2.RegionsSimilarityCalculator')

    similarity_calculator = region_similarity_calculator_config.WhichOneof(
        'region_similarity')
    if similarity_calculator == 'iou_similarity':
        return region_similarity_calculator.IouSimilarity()
    if similarity_calculator == 'ioa_similarity':
        return region_similarity_calculator.IoaSimilarity()
    if similarity_calculator == 'neg_sq_dist_similarity':
        return region_similarity_calculator.NegSqDistSimilarity()

    raise ValueError('Unknown region similarity calculator.')
