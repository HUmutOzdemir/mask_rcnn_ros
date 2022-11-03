from torch import nn
from typing import List
import torchvision.transforms as T
from torchvision.models.detection import (
    mask_rcnn,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)

from mask_rcnn_ros.config import COCO_CLASS_NAMES, CONFIG


class MaskRCNN(nn.Module):
    def __init__(
        self,
        version: str = "v1",
        config: dict = CONFIG,
        class_names: List[str] = COCO_CLASS_NAMES,
        mode: str = "eval",
        device: str = "cuda",
    ):
        super(MaskRCNN, self).__init__()

        assert mode in ["train", "eval"], "Possible Modes: [train, eval]"

        self.model = MaskRCNN._initialize_model(version, config)
        self.model.to(device)
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        self.device = device
        self.transform = T.Compose([T.ToTensor()])
        self.class_names = class_names

    @staticmethod
    def _initialize_model(version: str, config: dict):
        assert version in ["v1", "v2"], "Supported versions: [ v1, v2 ]"

        if version == "v1":
            return maskrcnn_resnet50_fpn(
                weights=mask_rcnn.MaskRCNN_ResNet50_FPN_Weights.DEFAULT, **config
            )
        else:
            return maskrcnn_resnet50_fpn_v2(
                weights=mask_rcnn.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, **config
            )

    def forward(self, images, targets=None):
        # Transform given image
        images = self.transform(images).to(self.device)
        # Expand tensor if it is necessary
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        # Run model
        return self.model(images)
