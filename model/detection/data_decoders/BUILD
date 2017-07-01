# DeepCV Object Detection API: data decoders.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])
# Apache 2.0

py_library(
    name = "tf_example_decoder",
    srcs = ["tf_example_decoder.py"],
    deps = [
        "//tensorflow",
        "//model/detection/core:data_decoder",
        "//model/detection/core:standard_fields",
    ],
)

py_test(
    name = "tf_example_decoder_test",
    srcs = ["tf_example_decoder_test.py"],
    deps = [
        ":tf_example_decoder",
        "//tensorflow",
        "//model/detection/core:standard_fields",
    ],
)
