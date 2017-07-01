# DeepCV Object Detection API: Meta-architectures.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_library(
    name = "ssd_meta_arch",
    srcs = ["ssd_meta_arch.py"],
    deps = [
        "//tensorflow",
        "//model/detection/core:box_coder",
        "//model/detection/core:box_list",
        "//model/detection/core:box_predictor",
        "//model/detection/core:model",
        "//model/detection/core:target_assigner",
        "//model/detection/utils:variables_helper",
    ],
)

py_test(
    name = "ssd_meta_arch_test",
    srcs = ["ssd_meta_arch_test.py"],
    deps = [
        ":ssd_meta_arch",
        "//tensorflow",
        "//tensorflow/python:training",
        "//model/detection/core:anchor_generator",
        "//model/detection/core:box_list",
        "//model/detection/core:losses",
        "//model/detection/core:post_processing",
        "//model/detection/core:region_similarity_calculator",
        "//model/detection/utils:test_utils",
    ],
)

py_library(
    name = "faster_rcnn_meta_arch",
    srcs = [
        "faster_rcnn_meta_arch.py",
    ],
    deps = [
        "//tensorflow",
        "//model/detection/anchor_generators:grid_anchor_generator",
        "//model/detection/core:balanced_positive_negative_sampler",
        "//model/detection/core:box_list",
        "//model/detection/core:box_list_ops",
        "//model/detection/core:box_predictor",
        "//model/detection/core:losses",
        "//model/detection/core:model",
        "//model/detection/core:post_processing",
        "//model/detection/core:standard_fields",
        "//model/detection/core:target_assigner",
        "//model/detection/utils:ops",
        "//model/detection/utils:variables_helper",
    ],
)

py_library(
    name = "faster_rcnn_meta_arch_test_lib",
    srcs = [
        "faster_rcnn_meta_arch_test_lib.py",
    ],
    deps = [
        ":faster_rcnn_meta_arch",
        "//tensorflow",
        "//model/detection/anchor_generators:grid_anchor_generator",
        "//model/detection/builders:box_predictor_builder",
        "//model/detection/builders:hyperparams_builder",
        "//model/detection/builders:post_processing_builder",
        "//model/detection/core:losses",
        "//model/detection/protos:box_predictor_py_pb2",
        "//model/detection/protos:hyperparams_py_pb2",
        "//model/detection/protos:post_processing_py_pb2",
    ],
)

py_test(
    name = "faster_rcnn_meta_arch_test",
    srcs = ["faster_rcnn_meta_arch_test.py"],
    deps = [
        ":faster_rcnn_meta_arch_test_lib",
    ],
)

py_library(
    name = "rfcn_meta_arch",
    srcs = ["rfcn_meta_arch.py"],
    deps = [
        ":faster_rcnn_meta_arch",
        "//tensorflow",
        "//model/detection/core:box_predictor",
        "//model/detection/utils:ops",
    ],
)

py_test(
    name = "rfcn_meta_arch_test",
    srcs = ["rfcn_meta_arch_test.py"],
    deps = [
        ":faster_rcnn_meta_arch_test_lib",
        ":rfcn_meta_arch",
        "//tensorflow",
    ],
)
