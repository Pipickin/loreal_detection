import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

def show_rectbox(args):
    """
    Shows images with selected objects in rectangles
    :param args: Parsed query which contains: images directory (input_dir),
    path to label map (label_map),path to frozen model (graph_dir),
    clasess number (num_classes)
    :return: Images with selected objects
    """

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, args['graph_dir'])

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, args['label_map'])

    # Path to directory of images
    PATH_TO_DIR = os.path.join(CWD_PATH, args['input_dir'])

    # Path to label map file
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

    # Set the categories connected with label map
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    for img in os.listdir(PATH_TO_DIR):
        if img.endswith('jpg'):
            # Expand dimension of image to get shape [1, None, None, 3]
            PATH_TO_IMAGE = os.path.join(PATH_TO_DIR, img)
            image = cv2.imread(PATH_TO_IMAGE)
            image_expanded = np.expand_dims(image, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            # Draw the results of the detection
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.60)

            cv2.imshow(img, image)
    cv2.waitKey(0)


if __name__ == '__main__':

    arg = argparse.ArgumentParser(description='Show rectbox on the images')
    arg.add_argument('--input_dir', type=str, required=True,
                     help='Directory of input images')
    arg.add_argument('--graph_dir', type=str, required=True,
                     help='Directory of graph')
    arg.add_argument('--label_map', type=str, required=True,
                     help='Directory of label map file')
    args = vars(arg.parse_args())

    show_rectbox(args)

    cv2.destroyAllWindows()
