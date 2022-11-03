import rospy
import threading
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, RegionOfInterest
from mask_rcnn_ros.msg import Result
from std_msgs.msg import Header

from mask_rcnn_ros.model import MaskRCNN
from mask_rcnn_ros.visualization import plot_result


class MaskRCNNROS:
    def __init__(self, version: str = "v1", device: str = "cuda") -> None:

        self._cv_bridge = CvBridge()

        # Get ROS Arguments.
        self._rgb_input_topic = rospy.get_param("~input", "/camera/rgb/image_raw")
        self._visualization = rospy.get_param("~visualization", True)
        self._publish_rate = rospy.get_param("~publish_rate", 100)

        # Initialize Model
        self.model = MaskRCNN(version, device)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        # Initialize Publishers and Listeners
        self._result_pub = rospy.Publisher("~result", RegionOfInterest, queue_size=1)
        self._vis_pub = rospy.Publisher("~visualization", Image, queue_size=1)
        rospy.Subscriber(
            self._rgb_input_topic, Image, self._image_callback, queue_size=1
        )

    def run(self):
        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():
            msg = self._last_message
            if msg:
                # Covert Image ROS Message to Numpy Array
                image = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                # Run MaskRCNN
                result = self.model(image)[0]
                # Publish Result Message
                self._result_pub.publish(self._build_result_msg(msg, result))

                # Visualize results
                if self._visualization:
                    cv_result = plot_result(image, result, True)
                    image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, "bgr8")
                    self._vis_pub.publish(image_msg)

            rate.sleep()

    def _build_result_msg(self, header: Header, result: dict) -> Result:
        msg = Result()
        msg.header = header

        for bbox, label, score, mask in zip(
            result["boxes"], result["labels"], result["scores"], result["masks"]
        ):
            msg.boxes.append(self._build_roi_msg(bbox))
            msg.class_ids.append(label.item())
            msg.class_names.append(self.model.class_names[label.item()])
            msg.scores.append(score.item())
            msg.masks.append(self._build_mask_msg(mask, header))

        return result

    def _build_roi_msg(self, bbox) -> RegionOfInterest:
        x1, y1, x2, y2 = bbox.cpu().detach().numpy()

        box = RegionOfInterest()

        box.x_offset = np.asscalar(x1)
        box.y_offset = np.asscalar(y1)
        box.height = np.asscalar(y2 - y1)
        box.width = np.asscalar(x2 - x1)

        return box

    def _build_mask_msg(self, mask, header: Header) -> Image:
        msg = Image()

        msg.header = header
        msg.height = mask.shape[1]
        msg.width = mask.shape[2]
        msg.encoding = "mono8"
        msg.is_bigendian = False
        msg.step = mask.shape[2]
        msg.data = mask.squeeze(0).cpu().detach().numpy().tobytes()

        return msg

    @property
    def _last_message(self):
        msg = None

        if self._msg_lock.acquire(False):
            msg = self._last_msg
            self._last_msg = None
            self._msg_lock.release()

        return msg

    def _image_callback(self, msg: Image):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()
