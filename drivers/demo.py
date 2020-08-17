from os import path

import tensorflow as tf

from models import build_detr_resnet50
from utils import read_jpeg, preprocess_image, plot_results
from chambers.utils.boxes import absolute2relative

# %%
detr = build_detr_resnet50()
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

# %%
image = read_jpeg(path.join('samples', 'sample_1.jpg'))

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
