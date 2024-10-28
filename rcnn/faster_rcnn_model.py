import torchvision
import torch

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import test


# 使用预训练的 ResNet50 作为主干网络
backbone = torchvision.models.resnet50(pretrained=True)
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # 移除最后的全连接层

# 定义 RPN 锚点生成器（不同尺度和纵横比的设置）
rpn_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

# 定义 RoI Pooling 层的特征映射尺寸
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# 创建 Faster R-CNN 模型
model = FasterRCNN(backbone,
                   num_classes=91,  # 根据目标检测的类别数设置（例如 COCO 数据集是 91 类）
                   rpn_anchor_generator=rpn_anchor_generator,
                   box_roi_pool=roi_pooler)

# 如果只想使用预训练的 Faster-RCNN with ResNet50 backbone
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)