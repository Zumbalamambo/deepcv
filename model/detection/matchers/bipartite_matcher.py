"""Bipartite matcher implementation."""

import tensorflow as tf
from tensorflow.contrib.image.python.ops import image_ops

import model.detection.core.matcher as matcher


class GreedyBipartiteMatcher(matcher.Matcher):
  """Wraps a Tensorflow greedy bipartite matcher."""

  def _match(self, similarity_matrix, num_valid_rows=-1):
    """Bipartite matches a collection rows and columns. A greedy bi-partite.

    TODO: Add num_valid_columns options to match only that many columns with
        all the rows.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher values mean more similar.
      num_valid_rows: A scalar or a 1-D tensor with one element describing the
        number of valid rows of similarity_matrix to consider for the bipartite
        matching. If set to be negative, then all rows from similarity_matrix
        are used.

    Returns:
      match_results: int32 tensor of shape [M] with match_results[i]=-1
        meaning that column i is not matched and otherwise that it is matched to
        row match_results[i].
    """
    # Convert similarity matrix to distance matrix as tf.image.bipartite tries
    # to find minimum distance matches.
    distance_matrix = -1 * similarity_matrix
    _, match_results = image_ops.bipartite_match(
        distance_matrix, num_valid_rows)
    match_results = tf.reshape(match_results, [-1])
    match_results = tf.cast(match_results, tf.int32)
    return match_results
