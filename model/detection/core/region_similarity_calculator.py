"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

import model.detection.core.box_list_ops as box_list_ops


class RegionSimilarityCalculator(object):
    """Abstract base class for region similarity calculator."""
    __metaclass__ = ABCMeta

    def compare(self, boxlist1, boxlist2, scope=None):
        """Computes matrix of pairwise similarity between BoxLists.

        This op (to be overriden) computes a measure of pairwise similarity between
        the boxes in the given BoxLists. Higher values indicate more similarity.

        Note that this method simply measures similarity and does not explicitly
        perform a matching.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.
          scope: Op scope name. Defaults to 'Compare' if None.

        Returns:
          a (float32) tensor of shape [N, M] with pairwise similarity score.
        """
        with tf.name_scope(scope, 'Compare', [boxlist1, boxlist2]) as scope:
            return self._compare(boxlist1, boxlist2)

    @abstractmethod
    def _compare(self, boxlist1, boxlist2):
        pass


class IouSimilarity(RegionSimilarityCalculator):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """

    def _compare(self, boxlist1, boxlist2):
        """Compute pairwise IOU similarity between the two BoxLists.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing pairwise iou scores.
        """
        return box_list_ops.iou(boxlist1, boxlist2)


class NegSqDistSimilarity(RegionSimilarityCalculator):
    """Class to compute similarity based on the squared distance metric.

    This class computes pairwise similarity between two BoxLists based on the
    negative squared distance metric.
    """

    def _compare(self, boxlist1, boxlist2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise squared distance.
        """
        return -1 * box_list_ops.sq_dist(boxlist1, boxlist2)


class IoaSimilarity(RegionSimilarityCalculator):
    """Class to compute similarity based on Intersection over Area (IOA) metric.

    This class computes pairwise similarity between two BoxLists based on their
    pairwise intersections divided by the areas of second BoxLists.
    """

    def _compare(self, boxlist1, boxlist2):
        """Compute pairwise IOA similarity between the two BoxLists.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing pairwise IOA scores.
        """
        return box_list_ops.ioa(boxlist1, boxlist2)
