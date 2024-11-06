from torch.utils.data import Dataset
import os
import config
import torch
from lxml import etree
from PIL import Image

class VOCDataset(Dataset):
  def __init__(self, voc_root, file_txt, transforms=None):
    self.root = voc_root
    self.image_dir = os.path.join(self.root, 'JPEGImages')
    self.annotation_dir = os.path.join(self.root, 'Annotations')
    self.transforms = transforms
    with open(os.path.join(self.root, file_txt), encoding='utf-8') as read:
      file_lists = [file.strip() for file in read.readlines()]
    # print(file_lists)
    # with open(file_lists, encoding='utf-8') as read:
    # file_lists = [line.strip() for line in read.readlines()]
    self.xml_lists = []
    self.jpg_lists = []
    for file in file_lists:
      xml_path = os.path.join(self.annotation_dir, file + '.xml')
      with open(xml_path, encoding='utf-8') as read:
        xml_str = read.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
      data = self.parse_xml_to_dict(xml)['annotation']
      if 'object' in data:
        self.xml_lists.append(xml_path)
        self.jpg_lists.append(os.path.join(self.image_dir, file + '.jpg'))
    self.class_dict = config.DefaultConfig().classes



  def __len__(self):
    return len(self.xml_lists)

  def __getitem__(self, idx):
    xml_path = self.xml_lists[idx]
    with open(xml_path, encoding='utf-8') as read:
      xml_str = read.read()
    xml = etree.fromstring(xml_str.encode('utf-8'))
    data = self.parse_xml_to_dict(xml)['annotation']
    # print(data)
    # img_path = os.path.join(self.image_dir, data['filename'])
    image = Image.open(self.jpg_lists[idx])
    if image.format != 'JPEG':
      raise ValueError('image format is not JPEG')
    boxes = []
    labels = []
    iscrowd = []

    for obj in data["object"]:
      xmin = float(obj["bndbox"]["xmin"])
      xmax = float(obj["bndbox"]["xmax"])
      ymin = float(obj["bndbox"]["ymin"])
      ymax = float(obj["bndbox"]["ymax"])

      # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
      if xmax <= xmin or ymax <= ymin:
          print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
          continue
      
      boxes.append([xmin, ymin, xmax, ymax])
      labels.append(self.class_dict[obj["name"]])
      if "difficult" in obj:
          iscrowd.append(int(obj["difficult"]))
      else:
          iscrowd.append(0)
    
    # print(boxes, labels)

    
    
    
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
    image_id = torch.tensor([idx])
    # print(boxes, labels)
    try:

      area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    except Exception as e:
      print(e)
      print(boxes)
      print(xml_path)

    target = {}
    target['boxes'] = boxes
    target['labels'] = labels
    target['image_id'] = image_id
    target['area'] = area
    target['iscrowd'] = iscrowd

    if self.transforms is not None:
      image, target = self.transforms(image, target)

    return image, target

  def get_height_and_width(self, idx):
    xml_path = self.xml_lists[idx]
    with open(xml_path) as read:
      xml_str = read.read()
    xml = etree.fromstring(xml_str.encode('utf-8'))
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
  
  def coco_index(self, idx):
    """
    该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
    由于不用去读取图片，可大幅缩减统计时间

    Args:
        idx: 输入需要获取图像的索引
    """
    # read xml
    xml_path = self.xml_lists[idx]
    with open(xml_path) as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str.encode('utf-8'))
    data = self.parse_xml_to_dict(xml)["annotation"]
    data_height = int(data["size"]["height"])
    data_width = int(data["size"]["width"])
    # img_path = os.path.join(self.img_root, data["filename"])
    # image = Image.open(img_path)
    # if image.format != "JPEG":
    #     raise ValueError("Image format not JPEG")
    boxes = []
    labels = []
    iscrowd = []
    for obj in data["object"]:
        xmin = float(obj["bndbox"]["xmin"])
        xmax = float(obj["bndbox"]["xmax"])
        ymin = float(obj["bndbox"]["ymin"])
        ymax = float(obj["bndbox"]["ymax"])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(self.class_dict[obj["name"]])
        iscrowd.append(int(obj["difficult"]))

    # convert everything into a torch.Tensor
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
    image_id = torch.tensor([idx])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    return (data_height, data_width), target

  @staticmethod
  def collate_fn(batch):
    return tuple(zip(*batch))