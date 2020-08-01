import matplotlib.pyplot as plt
import tensorflow as tf

from chambers.utils.boxes import box_xywh_to_xyxy, box_xyxy_to_yxyx, box_normalize_xyxy
from datasets import CocoDetection
from models import build_detr_resnet50
from utils import read_jpeg

COCO_PATH = "/home/crr/datasets/coco"
BATCH_SIZE = 2

# %%
# coco_data = CocoDetection(COCO_PATH, partition='train2017')
coco_data = CocoDetection(COCO_PATH, partition='val2017')  # boxes [x0, y0, w, h]
dataset = tf.data.Dataset.from_generator(lambda: coco_data, (tf.string, tf.int32, tf.float32, tf.int32))
dataset = dataset.map(lambda img_path, shape, boxes, labels: boxes)
# dataset = dataset.map(
#     lambda img_path, shape, boxes, labels: (read_jpeg(img_path), shape, box_xywh_to_xyxy(boxes), labels))

# dataset = dataset.map(lambda image, shape, boxes, labels: (*preprocess_image(image), shape, boxes, labels))
# dataset = dataset.padded_batch(batch_size=BATCH_SIZE,
#                                padded_shapes=((None, None, 3), (None, None), (2,), (None, 4), (None,)),
#                                padding_values=(
#                                    tf.constant(0.0), tf.constant(True), None, tf.constant(-1.), tf.constant(-1)))

dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(2))

it = iter(dataset)
#%%
b = next(it)
b[0]
b[1]
b.shape

# %%
# x, mask, shape, boxes, labels = next(it)
x, shape, boxes, labels = next(it)
print("X SHAPE:", x.shape)
print("IMG SHAPES:", shape)
img_h, img_w = shape

boxes  # [x0, y0, x1, y1]
boxes = box_normalize_xyxy(boxes, img_h, img_w)

# %%
boxes_xy = box_xyxy_to_yxyx(boxes)
colors = [[1.0, 0., 0.]]
box_img = tf.image.draw_bounding_boxes([x], [boxes_xy], colors)
box_img = tf.cast(box_img, tf.uint8)
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
