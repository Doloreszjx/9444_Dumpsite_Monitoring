import os
import datetime
import torch

from dataset import VOCDataset
# from backbone import resnet50_fpn_backbone
import config
import torchvision.transforms as transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from timer import Timer


default_config = config.DefaultConfig()
# train_data_set = VOCDataset('VOC2012', transforms=None, train=False)
# print(len(train_data_set))

# def create_model(num_classes):
#   backbone = resnet50_fpn_backbone(pretrain_path= default_config.backbone_path + 'resnet50.pth', trainable_layers=3)

#   return model
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
])
# transform = None
train_data_set = VOCDataset('VOC2012', transform=transform, train=True)
data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# 加载预训练的 Faster R-CNN 模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 获取输入特征的数量
in_features = model.roi_heads.box_predictor.cls_score.in_features

# 替换最后的分类头，类别数为21（20个类别+1个背景）
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)

# 定义优化器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 学习率调度器
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 训练模型
num_epochs = 15
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model.to(device)
timer = Timer()

for epoch in range(num_epochs):
    timer.start()
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 计算损失
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


    # 更新学习率
    lr_scheduler.step()
    timer.end()
    print(f"Epoch: {epoch}, Loss: {losses.item()}, takes: {timer}")
    save_files = {
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'lr_scheduler': lr_scheduler.state_dict(),
      'epoch': epoch}

    torch.save(save_files, "rcnn/save_weights/resNetFpn-model-{}.pth".format(epoch))