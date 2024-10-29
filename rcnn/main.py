import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import test


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(device)


dataset = test.CustomDataset(image_dir='/Users/xk/Desktop/unsw/9444/9444_Dumpsite_Monitoring/VOC2012/train/JPEGImages', annotation_dir='/Users/xk/Desktop/unsw/9444/9444_Dumpsite_Monitoring/VOC2012/train/Annotations', transform=ToTensor())
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
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


from torch.optim import SGD

# 将模型放到 GPU 上（如果有）
model.to(device)

# 定义优化器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch} finished, Loss: {losses.item()}")

    # 可以在每个 epoch 后进行评估