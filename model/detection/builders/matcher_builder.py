"""A function to build an object detection matcher from configuration."""

import model.detection.matchers.argmax_matcher as argmax_matcher
import model.detection.matchers.bipartite_matcher as bipartite_matcher

import model.detection.protos.matcher_pb2 as matcher_pb2


def build(matcher_config):
    """Builds a matcher object based on the matcher config.

    Args:
      matcher_config: A matcher.proto object containing the config for the desired
        Matcher.

    Returns:
      Matcher based on the config.

    Raises:
      ValueError: On empty matcher proto.
    """
    if not isinstance(matcher_config, matcher_pb2.Matcher):
        raise ValueError('matcher_config not of type matcher_pb2.Matcher.')
    if matcher_config.WhichOneof('matcher_oneof') == 'argmax_matcher':
        matcher = matcher_config.argmax_matcher
        matched_threshold = unmatched_threshold = None
        if not matcher.ignore_thresholds:
            matched_threshold = matcher.matched_threshold
            unmatched_threshold = matcher.unmatched_threshold
        return argmax_matcher.ArgMaxMatcher(
            matched_threshold=matched_threshold,
            unmatched_threshold=unmatched_threshold,
            negatives_lower_than_unmatched=matcher.negatives_lower_than_unmatched,
            force_match_for_each_row=matcher.force_match_for_each_row)
    if matcher_config.WhichOneof('matcher_oneof') == 'bipartite_matcher':
        return bipartite_matcher.GreedyBipartiteMatcher()
    raise ValueError('Empty matcher.')
