import json
class DefaultConfig:
  # data
  # image_dir = 'VOC2012/train/JPEGImages'
  # annotation_dir = 'VOC2012/train/Annotations'
  # voc_root = 'new_dumpsite_data'
  voc_root = 'new_dumpsite_data'
  model_name = 'rcnn/fasterrcnn_mobilenet_v3_large_320_fpn'

  with open('rcnn/classes.json', 'r') as f:
    classes = json.load(f)
