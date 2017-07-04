# DeepCV Object Detection API: Models.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_library(
    name = "feature_map_generators",
    srcs = [
        "feature_map_generators.py",
    ],
    deps = [
        "//tensorflow",
        "//utils/detection:ops",
    ],
)

py_test(
    name = "feature_map_generators_test",
    srcs = [
        "feature_map_generators_test.py",
    ],
    deps = [
        ":feature_map_generators",
        "//tensorflow",
    ],
)

py_library(
    name = "ssd_feature_extractor_test",
    srcs = [
        "ssd_feature_extractor_test.py",
    ],
    deps = [
        "//tensorflow",
    ],
)

py_library(
    name = "ssd_inception_v2_feature_extractor",
    srcs = [
        "ssd_inception_v2_feature_extractor.py",
    ],
    deps = [
        ":feature_map_generators",
        "//tensorflow",
        "//model/detection/meta_architectures:ssd_meta_arch",
        "//model/classification:inception_v2",
    ],
)

py_library(
    name = "ssd_mobilenet_v1_feature_extractor",
    srcs = ["ssd_mobilenet_v1_feature_extractor.py"],
    deps = [
        ":feature_map_generators",
        "//tensorflow",
        "//model/detection/meta_architectures:ssd_meta_arch",
        "//model/classification:mobilenet_v1",
    ],
)

py_test(
    name = "ssd_inception_v2_feature_extractor_test",
    srcs = [
        "ssd_inception_v2_feature_extractor_test.py",
    ],
    deps = [
        ":ssd_feature_extractor_test",
        ":ssd_inception_v2_feature_extractor",
        "//tensorflow",
    ],
)

py_test(
    name = "ssd_mobilenet_v1_feature_extractor_test",
    srcs = ["ssd_mobilenet_v1_feature_extractor_test.py"],
    deps = [
        ":ssd_feature_extractor_test",
        ":ssd_mobilenet_v1_feature_extractor",
        "//tensorflow",
    ],
)

py_library(
    name = "faster_rcnn_inception_resnet_v2_feature_extractor",
    srcs = [
        "faster_rcnn_inception_resnet_v2_feature_extractor.py",
    ],
    deps = [
        "//tensorflow",
        "//model/detection/meta_architectures:faster_rcnn_meta_arch",
        "//utils/detection:variables_helper",
        "//model/classification:inception_resnet_v2",
    ],
)

py_test(
    name = "faster_rcnn_inception_resnet_v2_feature_extractor_test",
    srcs = [
        "faster_rcnn_inception_resnet_v2_feature_extractor_test.py",
    ],
    deps = [
        ":faster_rcnn_inception_resnet_v2_feature_extractor",
        "//tensorflow",
    ],
)

py_library(
    name = "faster_rcnn_resnet_v1_feature_extractor",
    srcs = [
        "faster_rcnn_resnet_v1_feature_extractor.py",
    ],
    deps = [
        "//tensorflow",
        "//model/detection/meta_architectures:faster_rcnn_meta_arch",
        "//model/classification:resnet_utils",
        "//model/classification:resnet_v1",
    ],
)

py_test(
    name = "faster_rcnn_resnet_v1_feature_extractor_test",
    srcs = [
        "faster_rcnn_resnet_v1_feature_extractor_test.py",
    ],
    deps = [
        ":faster_rcnn_resnet_v1_feature_extractor",
        "//tensorflow",
    ],
)
