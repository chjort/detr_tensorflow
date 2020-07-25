from os import path

import matplotlib.pyplot as plt
import tensorflow as tf

from models import build_detr_resnet50
from utils import read_jpeg_image, preprocess_image, absolute2relative


def plot_results(img, labels, probs, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(img)
    ax = plt.gca()
    for cl, p, (xmin, ymin, xmax, ymax), c in zip(
            labels, probs, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{CLASSES[cl]}: {p:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()


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

image = read_jpeg_image(path.join('samples', 'sample_1.jpg'))

# %%
inp_image, mask = preprocess_image(image)
inp_image = tf.expand_dims(inp_image, axis=0)
mask = tf.expand_dims(mask, axis=0)
outputs = detr((inp_image, mask), post_process=True)
labels, scores, boxes = [outputs[k][0].numpy() for k in ['labels', 'scores', 'boxes']]

# %%
keep = scores > 0.7
labels = labels[keep]
scores = scores[keep]
boxes = boxes[keep]
boxes = absolute2relative(boxes, (image.shape[1], image.shape[0])).numpy()

plot_results(image.numpy(), labels, scores, boxes)
