import os
import functools
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from google.protobuf import text_format

import model.detection.trainer as trainer
import model.detection.builders.model_builder as model_builder
import model.detection.builders.input_reader_builder as input_reader_builder
import model.detection.protos.model_pb2 as model_pb2
import model.detection.protos.pipeline_pb2 as pipeline_pb2
import model.detection.protos.train_pb2 as train_pb2
import model.detection.protos.input_reader_pb2 as input_reader_pb2
import utils.detection.label_map_util as label_map_util
import utils.detection.visualization_utils as vis_util

slim = tf.contrib.slim


def run(config, args):
    if args.task == 'detect':
        if args.file is not None:
            detect_image_local(config, args)
        elif args.file_url is not None:
            detect_image_remote(config, args)
        else:
            print('No this file!')
    elif args.task == 'train':
        train_distributed(config, args)
    else:
        print('No this task!')


def detect_image_local(config, args):
    # parse arguments
    num_class = int(config.get('dataset', 'num_class'))
    label_path = config.get('dataset', 'label')
    ckpt_path = config.get('weights', 'pb')
    image_path = args.file

    # load file
    image_original = Image.open(image_path)
    img_width, img_height = image_original.size
    image_np = np.array(image_original.getdata()).reshape((img_height, img_width, 3)).astype(np.uint8)
    image = np.expand_dims(image_np, axis=0)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=num_class,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    IMAGE_SIZE = (12, 8)
    with detection_graph.as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True),
                        graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                feed_dict={image_tensor: image})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()


def detect_image_remote(config, args):
    pass


def train_distributed(config, args):
    train_dir = config.get('train', 'directory')
    num_clones = args.num_clones
    clone_on_cpu = args.clone_on_cpu

    if args.pipeline_config_path:
        model_config, train_config, input_config = get_configs_from_pipeline_file(args)
    else:
        model_config, train_config, input_config = get_configs_from_multiple_files(args)

    model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=True)

    create_input_dict_fn = functools.partial(input_reader_builder.build, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
        # Number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
        # Set up distributed training.
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                 job_name=task_info.type,
                                 task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    trainer.train(create_input_dict_fn, model_fn, train_config, master, task, num_clones, worker_replicas,
                  clone_on_cpu, ps_tasks, worker_job_name, is_chief, train_dir)


def get_configs_from_pipeline_file(args):
    """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

    Reads training config from file specified by pipeline_config_path flag.

    Returns:
      model_config: model_pb2.DetectionModel
      train_config: train_pb2.TrainConfig
      input_config: input_reader_pb2.InputReader
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(args.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    model_config = pipeline_config.model
    train_config = pipeline_config.train_config
    input_config = pipeline_config.train_input_reader

    return model_config, train_config, input_config


def get_configs_from_multiple_files(args):
    """Reads training configuration from multiple config files.

    Reads the training config from the following files:
      model_config: Read from --model_config_path
      train_config: Read from --train_config_path
      input_config: Read from --input_config_path

    Returns:
      model_config: model_pb2.DetectionModel
      train_config: train_pb2.TrainConfig
      input_config: input_reader_pb2.InputReader
    """
    train_config = train_pb2.TrainConfig()
    with tf.gfile.GFile(args.train_config_path, 'r') as f:
        text_format.Merge(f.read(), train_config)

    model_config = model_pb2.DetectionModel()
    with tf.gfile.GFile(args.model_config_path, 'r') as f:
        text_format.Merge(f.read(), model_config)

    input_config = input_reader_pb2.InputReader()
    with tf.gfile.GFile(args.input_config_path, 'r') as f:
        text_format.Merge(f.read(), input_config)

    return model_config, train_config, input_config
