import tensorflow as tf

from chambers.utils.boxes import box_xywh_to_xyxy
from chambers.augmentations import box_normalize_xyxy, resize
from datasets import CocoDetection
from models import build_detr_resnet50
from utils import read_jpeg, normalize_image

COCO_PATH = "/home/crr/datasets/coco"
BATCH_SIZE = 2

# %%
coco_data = CocoDetection(COCO_PATH, partition='val2017')  # boxes [x0, y0, w, h]
dataset = tf.data.Dataset.from_generator(lambda: coco_data, (tf.string, tf.float32, tf.int32))
dataset = dataset.map(lambda img_path, boxes, labels: (read_jpeg(img_path), box_xywh_to_xyxy(boxes), labels))
# dataset = dataset.map(lambda img, boxes, labels: (img, box_normalize_xyxy(boxes, img), labels))

# dataset = dataset.map(lambda image, shape, boxes, labels: (normalize_image(image), shape, boxes, labels))

# dataset = dataset.padded_batch(batch_size=BATCH_SIZE,
#                                padded_shapes=((None, None, 3), (None, None), (2,), (None, 4), (None,)),
#                                padding_values=(
#                                    tf.constant(0.0), tf.constant(True), None, tf.constant(-1.), tf.constant(-1)))

# dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(2))

it = iter(dataset)

# %%
x, boxes, labels = next(it)
print("X SHAPE:", x.shape)
print("BOXES SHAPE:", boxes.shape)
boxes

x.shape
xt, boxest = resize(x, boxes)

boxest_n = box_normalize_xyxy(boxest, xt)

#%%
import matplotlib.pyplot as plt

plt.imshow(tf.cast(resize(x, min_side=800.0, max_side=1333.0), tf.uint8).numpy())
plt.show()

# %%
import matplotlib.pyplot as plt
from chambers.utils.boxes import box_xyxy_to_yxyx

x_viz = xt
boxes_viz = boxest_n

boxes_viz = box_xyxy_to_yxyx(boxes_viz)
colors = [[1.0, 0., 0.]]
box_img = tf.image.draw_bounding_boxes([x_viz], [boxes_viz], colors)
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
