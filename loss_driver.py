from os import path

import matplotlib.pyplot as plt
import tensorflow as tf

from models import build_detr_resnet50
from utils import read_jpeg_image, preprocess_image, absolute2relative

# %%
detr = build_detr_resnet50()
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

# %%
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

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

#%%
coco_data = COCODatasetBBoxes(args.coco_path, partition='val2017', return_boxes=False)
dataset = tf.data.Dataset.from_generator(lambda: coco_data, (tf.int32, tf.string))
dataset = dataset.map(lambda img_id, img_path: (img_id, read_jpeg_image(img_path)))
dataset = dataset.map(lambda img_id, image: (img_id, *preprocess_image(image)))

dataset = dataset.padded_batch(batch_size=args.batch_size,
                               padded_shapes=((), (None, None, 3), (None, None)),
                               padding_values=(None, tf.constant(0.0), tf.constant(True)))
# %%
inp_image, mask = preprocess_image(image)
inp_image = tf.expand_dims(inp_image, axis=0)
mask = tf.expand_dims(mask, axis=0)

#%%
outputs = detr((inp_image, mask), post_process=False)
outputs["pred_logits"].shape
outputs["pred_boxes"].shape

#%%
outputs = detr((inp_image, mask), post_process=True)
labels, scores, boxes = [outputs[k][0].numpy() for k in ['labels', 'scores', 'boxes']]

# %%
keep = scores > 0.7
labels = labels[keep]
scores = scores[keep]
boxes = boxes[keep]
boxes = absolute2relative(boxes, (image.shape[1], image.shape[0])).numpy()
