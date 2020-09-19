from os import path

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class COCODatasetBBoxes(tf.keras.utils.Sequence):
    def __init__(self, cocopath, partition='val2017', return_boxes=True,
                 ignore_crowded=True, **kwargs):
        super().__init__(**kwargs)
        self.cocopath = cocopath
        self.partition = partition
        self.return_boxes = return_boxes
        self.ignore_crowded = ignore_crowded

        self.coco = COCO(path.join(cocopath, 'annotations',
                                   'instances_%s.json' % partition))
        self.img_ids = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.img_ids[idx])[0]
        img_path = path.join(self.cocopath, self.partition, img_info['file_name'])
        if not self.return_boxes:
            return self.img_ids[idx], img_path
        ann_ids = self.coco.getAnnIds(self.img_ids[idx])
        boxes = self.parse_annotations(ann_ids)
        return self.img_ids[idx], img_path, boxes

    def parse_annotations(self, ann_ids):
        boxes = []
        for ann in self.coco.loadAnns(ann_ids):
            if 'iscrowd' in ann and ann['iscrowd'] > 0 and self.ignore_crowded:
                continue
            box = ann['bbox'] + [ann['category_id']]
            box = np.array(box, dtype=np.float32)
            box[2:4] += box[0:2]
            boxes.append(box)
        return boxes


class CocoDetection(tf.keras.utils.Sequence):
    def __init__(self, cocopath, partition='val2017',
                 ignore_crowded=True, **kwargs):
        super().__init__(**kwargs)
        self.cocopath = cocopath
        self.partition = partition
        self.ignore_crowded = ignore_crowded

        self.coco = COCO(path.join(cocopath, 'annotations',
                                   'instances_%s.json' % partition))
        self.img_ids = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        ann_ids = self.coco.getAnnIds(self.img_ids[idx])

        img_info = self.coco.loadImgs(self.img_ids[idx])[0]
        img_path = path.join(self.cocopath, self.partition, img_info['file_name'])
        boxes, labels = self.parse_annotations(ann_ids)
        return img_path, boxes, labels

    def parse_annotations(self, ann_ids):
        boxes = []
        labels = []
        for ann in self.coco.loadAnns(ann_ids):
            if 'iscrowd' in ann and ann['iscrowd'] > 0 and self.ignore_crowded:
                continue
            box = ann['bbox']
            label = ann['category_id']
            boxes.append(box)
            labels.append(label)
        return boxes, labels
