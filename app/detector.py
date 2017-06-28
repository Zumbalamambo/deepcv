import json
from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
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
        pass
    else:
        print('No this task!')


def detect_image_local(config, args):
    # parse arguments
    num_class = int(config.get('weight', 'num_class'))
    label_path = config.get('weight', 'label')
    ckpt_path = config.get('weight', 'pb')
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
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=detection_graph) as sess:

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                feed_dict={image_tensor: image})
    result = []
    if args.print:
        pass

    if args.json:
        result = json.dumps(result)

    if args.show:
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

    return result


def detect_image_remote(config, args):
    pass


