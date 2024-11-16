"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json
import cv2
import torch
from tqdm import tqdm
import numpy as np

import transforms
import torchvision
from network_files import FasterRCNN
from backbone import resnet50_fpn_backbone
from dataset import VOCDataset
from torchvision.ops import box_iou
from train_utils import get_coco_api_from_dataset, CocoEvaluator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def create_res101_model(num_classes):

  backbone = resnet_fpn_backbone('resnet101', weights=None)
  model = FasterRCNN(backbone, num_classes=91)

  # get number of input features for the classifier
  # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  return model

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


# 定义计算 IoU 和 F1 Score 相关的函数
def calculate_metrics_for_boxes(pred_boxes, pred_labels, gt_boxes, gt_labels, num_classes, iou_threshold=0.5):
    ious = np.zeros(num_classes)
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    iou_counts = np.zeros(num_classes)

    for cls in range(num_classes):
        # 获取当前类别的预测框和真实框
        pred_cls_boxes = pred_boxes[pred_labels == cls]
        gt_cls_boxes = gt_boxes[gt_labels == cls]

        # 计算 IoU
        if len(pred_cls_boxes) > 0 and len(gt_cls_boxes) > 0:
            iou_matrix = box_iou(torch.tensor(pred_cls_boxes), torch.tensor(gt_cls_boxes))
            max_iou_per_pred, _ = iou_matrix.max(dim=1)  # 每个预测框的最大 IoU 值

            # 计算 TP, FP, FN
            tp[cls] = (max_iou_per_pred >= iou_threshold).sum().item()  # IoU >= 阈值的预测框数
            fp[cls] = len(pred_cls_boxes) - tp[cls]  # 预测框中不符合条件的数量
            fn[cls] = len(gt_cls_boxes) - tp[cls]    # 漏检的真实框数量

            # 累加 IoU（仅用于 mIoU 计算）和有效 IoU 计数
            ious[cls] = max_iou_per_pred.mean().item() if tp[cls] > 0 else 0
            iou_counts[cls] = 1 if tp[cls] > 0 else 0
        elif len(pred_cls_boxes) > 0:  # 只有预测框，无真实框
            fp[cls] = len(pred_cls_boxes)
        elif len(gt_cls_boxes) > 0:  # 只有真实框，无预测框
            fn[cls] = len(gt_cls_boxes)

    return ious, iou_counts, tp, fp, fn

def unnormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  mean = torch.tensor(mean)[:, None, None]
  std = torch.tensor(std)[:, None, None]
  return image * std + mean

# Functions for visualizing images and their bounding boxes
def visualize_sample(image, boxes, labels, label_map, file_name):
  fig, ax = plt.subplots(1)
  # Normalize image to the range [0, 1]
  image = unnormalize(image).permute(1, 2, 0)  # Convert image from (C, H, W) to (H, W, C)
  image = image / 255.0 if image.dtype == torch.uint8 else image.clip(0, 1)

  ax.imshow(image)

  for box, label in zip(boxes, labels):
    xmin, ymin, xmax, ymax = box
    width, height = xmax - xmin, ymax - ymin
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin - 5, label_map[label.item()], color='red', fontsize=10, backgroundcolor="white")

  plt.savefig('output/' + file_name + '.png')
  plt.close()

# 定义在图片上绘制预测结果的函数
# def draw_predictions(image, boxes, labels, scores):
#     class_names =  ["domestic garbage", "construction waste", "agriculture forestry","disposed garbage"]
#     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]  # BGR
#     image = np.array(image)  # 转换为 NumPy 数组
#     for box, label, score in zip(boxes, labels, scores):
#         color = colors[label % len(colors)]
#         x1, y1, x2, y2 = map(int, box)  # 将边界框坐标转换为整数
#         # 绘制边界框
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#         # 绘制类别标签和置信度
#         label_text = f"{class_names[label]}: {score:.2f}"
#         cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     return image
def compute_iou(box1, box2):

    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def main(parser_data):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # read class_indict
    label_json_path = 'rcnn/classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {v: k for k, v in class_dict.items()}

    VOC_root = 'rcnn_dumpsite_data/test'
    # check voc root
    # if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
    #     raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    with open('rcnn/classes.json', 'r') as json_file:
        label_map = json.load(json_file)
    label_name = {v : k for k, v in label_map.items()}

    # load validation data set
    val_dataset = VOCDataset(VOC_root, 'test.txt', data_transform["val"])
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=8,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    # 注意，这里的norm_layer要和训练脚本中保持一致
    # backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # model = FasterRCNN(backbone=backbone, num_classes=parser_data.num_classes + 1)
    model = create_res101_model(5)
    num_classes = parser_data.num_classes
    iou_sum = np.zeros(parser_data.num_classes)
    iou_count = np.zeros(parser_data.num_classes)
    tp_sum = np.zeros(num_classes)
    fp_sum = np.zeros(num_classes)
    fn_sum = np.zeros(num_classes)

    # 载入你自己训练好的模型权重
    weights_path = 'rcnn/save_weights/resNet101Fpn-model-9.pth'
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    weights_dict = torch.load(weights_path)
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)

    model.to(device)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    outputs_arr = []
    all_true_labels = []
    all_pred_labels = []
    all_true_boxes = []
    all_pred_boxes = []
    all_scores = []
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)      
            

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            for i in range(len(image)):
                pred_boxes = outputs[i]['boxes'].cpu().numpy()
                pred_labels = outputs[i]['labels'].cpu().numpy()
                scores = outputs[i]['scores'].cpu().numpy()
                keep = scores >= 0.5
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                scores = scores[keep]
                true_boxes = targets[i]['boxes'].numpy()
                true_labels = targets[i]['labels'].numpy()
                all_true_boxes.extend(true_boxes)
                all_true_labels.extend(true_labels)
                all_pred_boxes.extend(pred_boxes)
                all_pred_labels.extend(pred_labels)
                all_scores.extend(scores)
                    # 提取预测的边界框和标签
            for img, output, target in zip(image, outputs, targets):
                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                
                # 计算该批次的 IoU 和 F1 相关计数并累计
                batch_ious, batch_iou_counts, batch_tp, batch_fp, batch_fn = calculate_metrics_for_boxes(pred_boxes, pred_labels, gt_boxes, gt_labels, num_classes)
                iou_sum += batch_ious
                iou_count += batch_iou_counts
                tp_sum += batch_tp
                fp_sum += batch_fp
                fn_sum += batch_fn
                
                # visualize_sample(img.to(cpu_device), pred_boxes, pred_labels, label_name, target['name'])

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)
        unique_labels = np.unique(all_true_labels)
    results = {}

    for label in unique_labels:
        tp = 0
        fp = 0
        fn = 0
        
        label_true_boxes = [all_true_boxes[i] for i in range(len(all_true_labels)) if all_true_labels[i] == label]
        label_pred_boxes = [all_pred_boxes[i] for i in range(len(all_pred_labels)) if all_pred_labels[i] == label]

        for i, true_box in enumerate(label_true_boxes):
            best_iou = 0
            best_j = -1
            for j, pred_box in enumerate(label_pred_boxes):
                iou = compute_iou(true_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou > 0.5:
                tp += 1
                label_pred_boxes.pop(best_j)
            else:
                fn += 1
        
        fp = len(label_pred_boxes)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results[label] = {
            'Precision': precision,
            'Recall': recall,
        }
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]

    # 计算所有类别的 mIoU
    for label, metrics in results.items():
        print(f'Label {label_name[label]}: {metrics}')
    miou = np.sum(iou_sum) / np.sum(iou_count)
    print(f"Mean IoU (mIoU): {miou}")

    # 计算每个类别的 Precision, Recall, F1 Score
    precision = tp_sum / (tp_sum + fp_sum + 1e-6)  # 避免除零
    recall = tp_sum / (tp_sum + fn_sum + 1e-6)     # 避免除零
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # 避免除零
    mean_f1_score = np.nanmean(f1_score)  # 忽略 NaN 值
    print(f'recall: {recall}')
    print(f"F1 Score per class: {f1_score}")
    print(f"Mean F1 Score: {mean_f1_score}")
    # ious = coco_evaluator.coco_eval['ious']  # 获取每个类别的IoU
    # mean_ious = [np.mean([iou for iou in cat_iou if iou > 0]) for cat_iou in ious.values()]
    # mIoU = np.mean(mean_ious)
    # print("mIoU: {:.3f}".format(mIoU))
    # print(outputs_arr)
    # calculate COCO info for all classes
    # coco_stats, print_coco = summarize(coco_eval)

    # # calculate voc info for every classes(IoU=0.5)
    # voc_map_info_list = []
    # for i in range(len(category_index)):
    #     stats, _ = summarize(coco_eval, catId=i)
    #     voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    # print_voc = "\n".join(voc_map_info_list)
    # print(print_voc)
    # print(print_coco)
    # # 将验证结果保存至txt文件中
    # with open("record_mAP.txt", "w") as f:
    #     record_lines = ["COCO results:",
    #                     print_coco,
    #                     "",
    #                     "mAP(IoU=0.5) for each category:",
    #                     print_voc]
    #     f.write("\n".join(record_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    # parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数
    parser.add_argument('--num-classes', type=int, default='4', help='number of classes')

    # 数据集的根目录(VOCdevkit)
    # parser.add_argument('--data-path', default='/data/', help='dataset root')

    # 训练好的权重文件
    # parser.add_argument('--weights-path', default='./save_weights/model.pth', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when validation.')

    args = parser.parse_args()

    main(args)
