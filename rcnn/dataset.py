from torch.utils.data import Dataset
import os
import config
import torch
from lxml import etree
from PIL import Image

class VOCDataset(Dataset):
  def __init__(self, voc_root, transform=None, train=True):
    if train:
      self.root = os.path.join(voc_root, 'train')
      txt_file = os.path.join(self.root, 'train.txt')
    else:
      self.root = os.path.join(voc_root, 'test')
      txt_file = os.path.join(self.root, 'test.txt')
    self.image_dir = os.path.join(self.root, 'JPEGImages')
    self.annotation_dir = os.path.join(self.root, 'Annotations')
    self.transform = transform

    with open(txt_file) as read:
      self.xml_lists = [os.path.join(self.annotation_dir, line.strip() + '.xml') for line in read.readlines()]
    with open(txt_file) as read:  
      self.jpg_lists = [os.path.join(self.image_dir, line.strip() + '.jpg') for line in read.readlines()]
    self.class_dict = config.DefaultConfig().classes



  def __len__(self):
    return len(self.xml_lists)

  def __getitem__(self, idx):
    xml_path = self.xml_lists[idx]
    with open(xml_path) as read:
      xml_str = read.read()
    xml = etree.fromstring(xml_str)
    data = self.parse_xml_to_dict(xml)['annotation']
    # print(data)
    # img_path = os.path.join(self.image_dir, data['filename'])
    image = Image.open(self.jpg_lists[idx])
    if image.format != 'JPEG':
      raise ValueError('image format is not JPEG')
    boxes = []
    labels = []
    iscrowd = []
    for obj in data['object']:
      bbox = obj['bndbox']
      xmin = float(bbox['xmin'])
      ymin = float(bbox['ymin'])
      xmax = float(bbox['xmax'])
      ymax = float(bbox['ymax'])
      boxes.append([xmin, ymin, xmax, ymax])
      labels.append(self.class_dict[obj['name']])
      iscrowd.append(int(obj['difficult']))
    
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
    image_id = torch.tensor([idx])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    target = {}
    target['boxes'] = boxes
    target['labels'] = labels
    target['image_id'] = image_id
    target['area'] = area
    target['iscrowd'] = iscrowd

    if self.transform is not None:
      image = self.transform(image)

    return image, target

  def get_height_and_width(self, idx):
    xml_path = self.xml_lists[idx]
    with open(xml_path) as read:
      xml_str = read.read()
    xml = etree.fromstring(xml_str)
    data = self.parse_xml_to_dict(xml)['annotation']
    data_height = int(data['size']['height'])
    data_width = int(data['size']['width'])
    return data_height, data_width
  
  def parse_xml_to_dict(self, xml):
    
    if len(xml) == 0:
      return {xml.tag: xml.text}
    result = {}
    for child in xml:
      child_result = self.parse_xml_to_dict(child)
      if child.tag != 'object':
        result[child.tag] = child_result[child.tag]
      else:
        if child.tag not in result:
          result[child.tag] = []
        result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}