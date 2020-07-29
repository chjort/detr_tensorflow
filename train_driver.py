import tensorflow as tf

from chambers.utils.boxes import boxes_resize_guard, normalize_boxes, box_cxcywh_to_yxyx
from datasets import CocoDetection
from models import build_detr_resnet50
from utils import read_jpeg_image, preprocess_image

COCO_PATH = "/home/crr/datasets/coco"
BATCH_SIZE = 2

# %%
coco_data = CocoDetection(COCO_PATH, partition='train2017')
dataset = tf.data.Dataset.from_generator(lambda: coco_data, (tf.string, tf.int32, tf.float32, tf.int32))
# dataset = dataset.map(lambda img_path, shape, boxes, labels: (read_jpeg_image(img_path), shape, boxes_resize_guard(boxes, shape[0], shape[1]), labels))
dataset = dataset.map(lambda img_path, shape, boxes, labels: (read_jpeg_image(img_path), shape, boxes, labels))
dataset = dataset.map(lambda image, shape, boxes, labels: (*preprocess_image(image), shape, boxes, labels))
dataset = dataset.padded_batch(batch_size=BATCH_SIZE,
                               padded_shapes=((None, None, 3), (None, None), (2,), (None, 4), (None,)),
                               padding_values=(
                                   tf.constant(0.0), tf.constant(True), None, tf.constant(-1.), tf.constant(-1)))

it = iter(dataset)
# x, mask, shape, boxes, labels = next(it)
x, shape, boxes, labels = next(it)
print("X SHAPE:", x.shape)
print("IMG SHAPES:", shape)
# boxes: [x, y, w + x, h + y]

shape
boxes
x.shape

# %%
import matplotlib.pyplot as plt

plt.imshow(x.numpy())
plt.show()

boxes_norm = normalize_boxes(boxes, x.shape[0], x.shape[1])
boxes_norm = box_cxcywh_to_yxyx(boxes_norm)
colors = [[1.0, 0., 0.]]
box_img = tf.cast(tf.image.draw_bounding_boxes([x], [boxes_norm], colors), tf.uint8)
box_img = box_img[0]

plt.imshow(box_img.numpy())
plt.show()

# %%
detr = build_detr_resnet50()
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

# %%
outputs = detr((x, mask), post_process=False)

print("PRED LOGITS:", outputs["pred_logits"].shape)
print("PRED BOXES:", outputs["pred_boxes"].shape)
