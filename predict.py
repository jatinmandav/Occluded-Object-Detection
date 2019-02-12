import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from yolo_prediction import *

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class CoffeeCupConfig(Config):
    """Configuration for training on the Coffee Cup dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "coffee_cup"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + coffee_cup

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

def form_mask(image, mask):
    image, result = make_prediction(image)

    update_mask = np.zeros(mask.shape, dtype=np.bool)
    for r in result:
        update_mask[r[2]:r[4], r[1]:r[3]] = mask[r[2]:r[4], r[1]:r[3]]

    mask = update_mask[:]

    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    color = np.empty(image.shape)
    color[:] = [200, 0, 0]
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, color, image).astype(np.uint8)
    else:
        splash = image.astype(np.uint8)

    return splash

def detect_and_color_splash(model, image_path=None):
    import cv2
    # Run model detection and generate the color splash effect
    print("Running on {}".format(args.image))
    # Read image
    image = cv2.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    print(r['scores'], r['class_ids'], r['rois'])
    # Color splash
    splash = form_mask(image, r['masks'])
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = [255, 0, 0]
    cv2.putText(splash, '{}:{}'.format('Mask Score', r['scores'][0]),
            (10, 20), font, 0.6, color, 2)

    cv2.imshow('result', splash)
    cv2.waitKey(0)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Detect coffee_cups with MaskRCNN+YOLO Network')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=True,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments

    print("Weights: ", args.weights)
    logs = 'logs/'

    # Configurations
    class InferenceConfig(CoffeeCupConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config,
                            model_dir=logs)

    weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    image = cv2.imread(args.image)
    cv2.imshow('Original Image', image)
    detect_and_color_splash(model, image_path=args.image)
