import json
class DefaultConfig:
  # data
  # image_dir = 'VOC2012/train/JPEGImages'
  # annotation_dir = 'VOC2012/train/Annotations'
  voc_root = 'data_balanced'
  with open('rcnn/classes.json', 'r') as f:
    classes = json.load(f)
