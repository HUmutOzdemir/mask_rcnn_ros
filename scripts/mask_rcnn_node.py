#!/usr/bin/env python
import rospy
import torch
from mask_rcnn_ros.ros_bridge import MaskRCNNROS

MASK_RCNN_VERSION = "v1"


def main():
    rospy.init_node("mask_rcnn")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing Mask R-CNN...")
    node = MaskRCNNROS(MASK_RCNN_VERSION, device=device)
    print("Mask R-CNN is Initialized...")
    node.run()


if __name__ == "__main__":
    main()
