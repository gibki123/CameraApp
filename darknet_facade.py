import os
import pathlib
import cv2
import sys

sys.path.append("../../darknet/")
import darknet as dn
import darknet_images as dimgs

NETWORK = None
CLASS_NAMES = None
COLORS = None
PROJECT_PATH = None
BATCH_SIZE = 1


def load_darknet():
    global NETWORK, CLASS_NAMES, COLORS, PROJECT_PATH, BATCH_SIZE
    PROJECT_PATH = pathlib.Path().absolute()
    config_path = os.path.join(PROJECT_PATH,'obj.cfg')
    data_path = os.path.join(PROJECT_PATH,'obj.data')
    weights_path = os.path.join(PROJECT_PATH,'weights','resnet_6000.weights')
    NETWORK, CLASS_NAMES, COLORS = dn.load_network(
                                                config_file = config_path,
                                                data_file = data_path,
                                                weights = weights_path,
                                                batch_size = BATCH_SIZE)
    return dn.network_width(NETWORK), dn.network_width(NETWORK)


def detect_candles(image_name):
    image_path = os.path.join(PROJECT_PATH, image_name)
    image, detections = dimgs.image_detection(image_path, NETWORK, CLASS_NAMES,
                                           COLORS, thresh = 0.15)
    cv2.imwrite(f'output_screen_candle.png', image)
    return detections

def make_image(a,b,c):
        return dn.make_image(a,b,c)
