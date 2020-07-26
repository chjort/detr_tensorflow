import tensorflow as tf

from datasets import CocoDetection
from models import build_detr_resnet50
from utils import read_jpeg_image, preprocess_image

COCO_PATH = "/home/crr/datasets/coco"
BATCH_SIZE = 2

# %%
coco_data = CocoDetection(COCO_PATH, partition='train2017')
dataset = tf.data.Dataset.from_generator(lambda: coco_data, (tf.string, tf.float32))
dataset = dataset.map(lambda img_path, boxes: (read_jpeg_image(img_path), boxes))
dataset = dataset.map(lambda image, boxes: (*preprocess_image(image), boxes))
dataset = dataset.padded_batch(batch_size=BATCH_SIZE,
                               padded_shapes=((None, None, 3), (None, None), (None, 4)),
                               padding_values=(tf.constant(0.0), tf.constant(True), tf.constant(-1.)))
it = iter(dataset)
x, mask, boxes = next(it)
print("X SHAPE:", x.shape)

# boxes: [x0, y0, x1, y1]

# %%
detr = build_detr_resnet50()
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

# %%
outputs = detr((x, mask), post_process=False)

print("PRED LOGITS:", outputs["pred_logits"].shape)
print("PRED BOXES:", outputs["pred_boxes"].shape)
