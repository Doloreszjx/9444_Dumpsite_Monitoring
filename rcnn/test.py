from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os
import torch

class CustomDataset(Dataset):
  def __init__(self, image_dir, annotation_dir, transform=None):
    self.image_dir = image_dir
    self.annotation_dir = annotation_dir
    self.transform = transform
    self.images = os.listdir(image_dir)

  def __getitem__(self, index):
    img_path = os.path.join(self.image_dir, self.images[index])
    img = Image(img_path).convert('RGB')

    annotation_path = os.path.join(self.annotation_dir, self.images[index].replace('.jpg', '.xml'))

    boxes = []
    labels = []

    taget = {
      'boxes': torch.tensor(boxes, dtype=torch.float32),
      'labels': torch.tensor(labels, dtype=torch.int64)
    }

    if self.transform:
      img, target = self.transform(img, target)
    
    return img, target
  def __len__(self):
    return len(self.images)

# demo
# dataset = CustomDataset(image_dir='/Users/xk/Desktop/unsw/9444/9444_Dumpsite_Monitoring/VOC2012/train/JPEGImages', annotation_dir='/Users/xk/Desktop/unsw/9444/9444_Dumpsite_Monitoring/VOC2012/train/Annotations', transform=ToTensor())
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))