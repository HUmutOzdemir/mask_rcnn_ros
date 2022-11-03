#!/usr/bin/env python
import rospy
import torch
from mask_rcnn_node.ros_bridge import MaskRCNNROS

MASK_RCNN_VERSION = "v1"


def main():
    rospy.init_node("mask_rcnn")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    node = MaskRCNNROS(MASK_RCNN_VERSION, device=device)
    node.run()


if __name__ == "__main__":
    main()
