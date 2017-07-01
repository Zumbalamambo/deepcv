# DeepCV Object Detection API: component builders.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0
py_library(
    name = "model_builder",
    srcs = ["model_builder.py"],
    deps = [
        ":anchor_generator_builder",
        ":box_coder_builder",
        ":box_predictor_builder",
        ":hyperparams_builder",
        ":image_resizer_builder",
        ":losses_builder",
        ":matcher_builder",
        ":post_processing_builder",
        ":region_similarity_calculator_builder",
        "//model/detection/core:box_predictor",
        "//model/detection/meta_architectures:faster_rcnn_meta_arch",
        "//model/detection/meta_architectures:rfcn_meta_arch",
        "//model/detection/meta_architectures:ssd_meta_arch",
        "//model/detection/models:faster_rcnn_inception_resnet_v2_feature_extractor",
        "//model/detection/models:faster_rcnn_resnet_v1_feature_extractor",
        "//model/detection/models:ssd_inception_v2_feature_extractor",
        "//model/detection/models:ssd_mobilenet_v1_feature_extractor",
        "//model/detection/protos:model_py_pb2",
    ],
)

py_test(
    name = "model_builder_test",
    srcs = ["model_builder_test.py"],
    deps = [
        ":model_builder",
        "//tensorflow",
        "//model/detection/meta_architectures:faster_rcnn_meta_arch",
        "//model/detection/meta_architectures:ssd_meta_arch",
        "//model/detection/models:ssd_inception_v2_feature_extractor",
        "//model/detection/models:ssd_mobilenet_v1_feature_extractor",
        "//model/detection/protos:model_py_pb2",
    ],
)

py_library(
    name = "matcher_builder",
    srcs = ["matcher_builder.py"],
    deps = [
        "//model/detection/matchers:argmax_matcher",
        "//model/detection/matchers:bipartite_matcher",
        "//model/detection/protos:matcher_py_pb2",
    ],
)

py_test(
    name = "matcher_builder_test",
    srcs = ["matcher_builder_test.py"],
    deps = [
        ":matcher_builder",
        "//model/detection/matchers:argmax_matcher",
        "//model/detection/matchers:bipartite_matcher",
        "//model/detection/protos:matcher_py_pb2",
    ],
)

py_library(
    name = "box_coder_builder",
    srcs = ["box_coder_builder.py"],
    deps = [
        "//model/detection/box_coders:faster_rcnn_box_coder",
        "//model/detection/box_coders:mean_stddev_box_coder",
        "//model/detection/box_coders:square_box_coder",
        "//model/detection/protos:box_coder_py_pb2",
    ],
)

py_test(
    name = "box_coder_builder_test",
    srcs = ["box_coder_builder_test.py"],
    deps = [
        ":box_coder_builder",
        "//tensorflow",
        "//model/detection/box_coders:faster_rcnn_box_coder",
        "//model/detection/box_coders:mean_stddev_box_coder",
        "//model/detection/box_coders:square_box_coder",
        "//model/detection/protos:box_coder_py_pb2",
    ],
)

py_library(
    name = "anchor_generator_builder",
    srcs = ["anchor_generator_builder.py"],
    deps = [
        "//model/detection/anchor_generators:grid_anchor_generator",
        "//model/detection/anchor_generators:multiple_grid_anchor_generator",
        "//model/detection/protos:anchor_generator_py_pb2",
    ],
)

py_test(
    name = "anchor_generator_builder_test",
    srcs = ["anchor_generator_builder_test.py"],
    deps = [
        ":anchor_generator_builder",
        "//tensorflow",
        "//model/detection/anchor_generators:grid_anchor_generator",
        "//model/detection/anchor_generators:multiple_grid_anchor_generator",
        "//model/detection/protos:anchor_generator_py_pb2",
    ],
)

py_library(
    name = "input_reader_builder",
    srcs = ["input_reader_builder.py"],
    deps = [
        "//tensorflow",
        "//model/detection/data_decoders:tf_example_decoder",
        "//model/detection/protos:input_reader_py_pb2",
    ],
)

py_test(
    name = "input_reader_builder_test",
    srcs = [
        "input_reader_builder_test.py",
    ],
    deps = [
        ":input_reader_builder",
        "//tensorflow",
        "//model/detection/core:standard_fields",
        "//model/detection/protos:input_reader_py_pb2",
    ],
)

py_library(
    name = "losses_builder",
    srcs = ["losses_builder.py"],
    deps = [
        "//model/detection/core:losses",
        "//model/detection/protos:losses_py_pb2",
    ],
)

py_test(
    name = "losses_builder_test",
    srcs = ["losses_builder_test.py"],
    deps = [
        ":losses_builder",
        "//model/detection/core:losses",
        "//model/detection/protos:losses_py_pb2",
    ],
)

py_library(
    name = "optimizer_builder",
    srcs = ["optimizer_builder.py"],
    deps = [
        "//tensorflow",
        "//model/detection/utils:learning_schedules",
    ],
)

py_test(
    name = "optimizer_builder_test",
    srcs = ["optimizer_builder_test.py"],
    deps = [
        ":optimizer_builder",
        "//tensorflow",
        "//model/detection/protos:optimizer_py_pb2",
    ],
)

py_library(
    name = "post_processing_builder",
    srcs = ["post_processing_builder.py"],
    deps = [
        "//tensorflow",
        "//model/detection/core:post_processing",
        "//model/detection/protos:post_processing_py_pb2",
    ],
)

py_test(
    name = "post_processing_builder_test",
    srcs = ["post_processing_builder_test.py"],
    deps = [
        ":post_processing_builder",
        "//tensorflow",
        "//model/detection/protos:post_processing_py_pb2",
    ],
)

py_library(
    name = "hyperparams_builder",
    srcs = ["hyperparams_builder.py"],
    deps = [
        "//model/detection/protos:hyperparams_py_pb2",
    ],
)

py_test(
    name = "hyperparams_builder_test",
    srcs = ["hyperparams_builder_test.py"],
    deps = [
        ":hyperparams_builder",
        "//tensorflow",
        "//model/detection/protos:hyperparams_py_pb2",
    ],
)

py_library(
    name = "box_predictor_builder",
    srcs = ["box_predictor_builder.py"],
    deps = [
        ":hyperparams_builder",
        "//model/detection/core:box_predictor",
        "//model/detection/protos:box_predictor_py_pb2",
    ],
)

py_test(
    name = "box_predictor_builder_test",
    srcs = ["box_predictor_builder_test.py"],
    deps = [
        ":box_predictor_builder",
        ":hyperparams_builder",
        "//tensorflow",
        "//model/detection/protos:box_predictor_py_pb2",
        "//model/detection/protos:hyperparams_py_pb2",
    ],
)

py_library(
    name = "region_similarity_calculator_builder",
    srcs = ["region_similarity_calculator_builder.py"],
    deps = [
        "//model/detection/core:region_similarity_calculator",
        "//model/detection/protos:region_similarity_calculator_py_pb2",
    ],
)

py_test(
    name = "region_similarity_calculator_builder_test",
    srcs = ["region_similarity_calculator_builder_test.py"],
    deps = [
        ":region_similarity_calculator_builder",
        "//tensorflow",
    ],
)

py_library(
    name = "preprocessor_builder",
    srcs = ["preprocessor_builder.py"],
    deps = [
        "//tensorflow",
        "//model/detection/core:preprocessor",
        "//model/detection/protos:preprocessor_py_pb2",
    ],
)

py_test(
    name = "preprocessor_builder_test",
    srcs = [
        "preprocessor_builder_test.py",
    ],
    deps = [
        ":preprocessor_builder",
        "//tensorflow",
        "//model/detection/core:preprocessor",
        "//model/detection/protos:preprocessor_py_pb2",
    ],
)

py_library(
    name = "image_resizer_builder",
    srcs = ["image_resizer_builder.py"],
    deps = [
        "//tensorflow",
        "//model/detection/core:preprocessor",
        "//model/detection/protos:image_resizer_py_pb2",
    ],
)

py_test(
    name = "image_resizer_builder_test",
    srcs = ["image_resizer_builder_test.py"],
    deps = [
        ":image_resizer_builder",
        "//tensorflow",
        "//model/detection/protos:image_resizer_py_pb2",
    ],
)
