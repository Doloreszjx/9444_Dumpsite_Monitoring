from dataset import VOCDataset
import transforms
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import config

default_config = config.DefaultConfig()
# 设置测试数据的 transforms
transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)])

# 加载测试数据集
test_dataset = VOCDataset(default_config.voc_root, transforms=transforms, train=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# 加载 Faster R-CNN 模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 7  # VOC 数据集通常有 20 类目标 + 背景
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# 加载训练好的模型参数
model.load_state_dict(torch.load('rcnn/save_weights/resNetFpn-model-balanced-7.pth')['model'])
# model.eval()

def compute_iou(box1, box2):
    # box1, box2 格式：[x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    return iou

def evaluate_model(model, test_loader, iou_threshold=0.5):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    true_positives, false_positives, false_negatives = 0, 0, 0
    # device = 'cpu'
    
    print(f'Using device {device}')
    for images, targets in tqdm(test_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)
        iou_list = []
        # print(targets)

        for output, target in zip(outputs, targets):
            pred_boxes = output['boxes'].cpu().detach().numpy()
            true_boxes = target['boxes'].cpu().detach().numpy()

            for pred_box in pred_boxes:
                if any(compute_iou(pred_box, true_box) >= iou_threshold for true_box in true_boxes):
                    true_positives += 1
                else:
                    false_positives += 1

            for true_box in true_boxes:
                if not any(compute_iou(true_box, pred_box) >= iou_threshold for pred_box in pred_boxes):
                    false_negatives += 1
            for pred_box in pred_boxes:
                    ious = [compute_iou(pred_box, true_box) for true_box in true_boxes]
                    max_iou = max(ious) if ious else 0  # 选择最佳匹配 IoU
                    iou_list.append(max_iou)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
    miou = np.mean(iou_list) if iou_list else 0
    print(f"Mean IoU (mIoU) for test set: {miou:.4f}")
    return miou, precision, recall

evaluate_model(model, test_loader, iou_threshold=0.5)