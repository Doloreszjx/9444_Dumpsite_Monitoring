import os
import datetime
import torch

from dataset import VOCDataset
# from backbone import resnet50_fpn_backbone
import config
import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from train_utils import train_eval_utils as utils




def create_resnet50_model(num_classes):

    # get number of input features for the classifier
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model




if __name__ == "__main__":
  default_config = config.DefaultConfig()
  model = create_resnet50_model(num_classes=5)
  transforms = {
    "train": transforms.Compose([transforms.ToTensor(),
                                  transforms.RandomHorizontalFlip(0.5),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  }
  train_data_set = VOCDataset(os.path.join(default_config.voc_root, 'train'), 'train.txt', transforms=transforms['train'])
  train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
  val_dataset = VOCDataset(os.path.join(default_config.voc_root, 'test'), "test.txt", transforms=transforms["val"])
  val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=4,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      collate_fn=lambda x: tuple(zip(*x)))
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
  results_file = "rcnn/coco_info/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


  # 训练模型
  num_epochs = 10
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  model.to(device)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


  train_loss = []
  learning_rate = []
  val_map = []
  res50_losses = []
  epoch_time = []
  for epoch in range(0, num_epochs):
    # train for one epoch, printing every 10 iterations
    mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                          device=device, epoch=epoch,
                                          print_freq=50, warmup=True,
                                          scaler=None)
    train_loss.append(mean_loss.item())
    learning_rate.append(lr)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    coco_info = utils.evaluate(model, val_data_set_loader, device=device)
    res50_losses.append(mean_loss.item())
    epoch_time.append(epoch)
    print(f"epoch:{epoch} loss{mean_loss.item()}")
    # write into txt
    with open(results_file, "a") as f:
        # 写入的数据包括coco指标还有loss和learning rate
        result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
        txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        f.write(txt + "\n")

    val_map.append(coco_info[1])  # pascal mAP

    # save weights
    save_files = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch}
    torch.save(save_files, "rcnn/save_weights/resNet50Fpn-model-{}.pth".format(epoch))

