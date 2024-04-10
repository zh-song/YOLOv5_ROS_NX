#!/usr/bin/env python3
# -*-coding:utf-8-*

import cv2
import argparse
import torch
import rospy
import sys
import logging
import time
import numpy as np
import math
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import yaml
from copy import deepcopy
from detectAPI import YoLov5TRT
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
from swarmpolicy.msg import BBox
from swarmpolicy.msg import DetResults

from utils.general import set_logging, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.datasets import letterbox
from obj_cfg import detect_cfg


sys.path.append('/')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

def callback(data):
    global count, bridge, config, results_publisher, frame_seq
    count = count + 1
    if count == 1:
        count = 0
        cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
        img_msg_pack = det.infer(cv_img)
        det.destroy()

        frame_seq = int(data.header.frame_id)
        results = DetResults()
        results.frame_id = frame_seq

        # 当检测无目标时发布空目标
        if len(img_msg_pack) == 0:
            result = BBox()
            result.seq = -1
            result.bbox = []
            result.camera_id = 0
            results.det_results.append(result)

        # 当检测出目标时直接发布检测结果
        for det_result in img_msg_pack:
            result = BBox()
            result.seq = 0
            result.camera_id = 1
            result.bbox = det_result[0:]
            results.det_results.append(result)

        results_publisher.publish(results)
        # ADD LOG
        rospy.loginfo("%d frame: got target %d", results.frame_id, len(img_msg_pack))


if __name__ == '__main__':
    rospy.init_node("TensoRTDetect", anonymous=False)
    print("THIS NODE USED TO DETECT CURRENT FRAME!!!")
    opt = detect_cfg

    global count, bridge, det, config, results_publisher, frame_seq
    frame_seq = 0
    count = 0
    bridge = CvBridge()
    det = YoLov5TRT(opt)
    # Hyperparameters
    with open(opt.config) as f:
        config = yaml.safe_load(f)  # load hyps

    # Publish detect results to other Nodes.
    results_publisher = rospy.Publisher("Detect_results", DetResults, queue_size=1)
    # Get Frame
    rospy.Subscriber('Frame', Image, callback, queue_size=1, buff_size=10000000)

    rospy.spin()
