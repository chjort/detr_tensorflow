import pickle

import numpy as np
import tensorflow as tf

from chambers.utils.boxes import box_cxcywh_to_xyxy, boxes_giou
from chambers.utils.masking import remove_padding_box, remove_padding_1d
from chambers.utils.tf import pairwise_l1
from chambers.utils.tf import linear_sum_assignment, sizes_to_batch_indices, batch_linear_sum_assignment

COCO_PATH = "/home/crr/datasets/coco"
# COCO_PATH = "/home/ch/datasets/coco"
BATCH_SIZE = 2

# %%
samples = np.load("x_pt.npy")
samples = tf.convert_to_tensor(samples)
samples = tf.transpose(samples, [0, 2, 3, 1])

mask = np.load("mask_pt.npy")
mask = tf.convert_to_tensor(mask)

x1 = tf.where(tf.expand_dims(mask[0], -1), tf.ones_like(samples[0]) * -1., samples[0])  # masked values to -1
x2 = tf.where(tf.expand_dims(mask[1], -1), tf.ones_like(samples[1]) * -1., samples[1])  # masked values to -1
x = tf.stack([x1, x2], axis=0)

with open("y_pt.pickle", "rb") as f:
    targets = pickle.load(f)
    targets = [{k: tf.convert_to_tensor(v) for k, v in t.items() if k in ("boxes", "labels")} for t in targets]

with open("pred_pt.pickle", "rb") as f:
    pred = pickle.load(f)
    pred["pred_logits"] = tf.convert_to_tensor(pred["pred_logits"])
    pred["pred_boxes"] = tf.convert_to_tensor(pred["pred_boxes"])
    pred["aux_outputs"] = [{k: tf.convert_to_tensor(v) for k, v in aux_output.items()}
                           for aux_output in pred["aux_outputs"]]
    class_out = pred["pred_logits"]
    box_out = pred["pred_boxes"]

boxes = [target["boxes"] for target in targets]
labels = [target["labels"] for target in targets]

boxes1 = tf.pad(boxes[0], paddings=[[0, 22], [0, 0]], constant_values=-1.)
boxes2 = boxes[1]
boxes = tf.stack([boxes1, boxes2], axis=0)

labels1 = tf.pad(labels[0], paddings=[[0, 22]], constant_values=-1)
labels2 = labels[1]
labels = tf.cast(tf.stack([labels1, labels2], axis=0), tf.int32)

# %%
x.shape
boxes.shape
labels.shape

class_out.shape  # shape (batch_size, n_boxes, n_classes)
box_out.shape  # shape (batch_size, n_boxes, 4), box coordinates (center_x, center_y, w, h)

# %% Matcher
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2

# stacking batches
batch_size, n_box, n_class = class_out.shape
box_shape = box_out.shape[-1]

y_pred_logits = tf.reshape(class_out, [-1, n_class])  # out_bbox
y_pred_logits = tf.nn.softmax(y_pred_logits, axis=-1)
y_true_logits = tf.reshape(labels, [-1])  # tgt_pred
y_true_logits = remove_padding_1d(y_true_logits, -1)

y_pred_box = tf.reshape(box_out, [-1, box_shape])  # out_bbox
y_true_box = tf.reshape(boxes, [-1, tf.shape(boxes)[-1]])
y_true_box = remove_padding_box(y_true_box, -1.)

# bbox cost
cost_bbox = pairwise_l1(y_pred_box, y_true_box)

# logits cost
cost_class = -tf.gather(y_pred_logits, y_true_logits, axis=1)

# giou cost
y_true_box = box_cxcywh_to_xyxy(y_true_box)
y_pred_box = box_cxcywh_to_xyxy(y_pred_box)
cost_giou = -boxes_giou(y_pred_box, y_true_box)

# Weighted cost matrix
C = set_cost_bbox * cost_bbox + set_cost_class * cost_class + set_cost_giou * cost_giou
C = tf.reshape(C, [batch_size, n_box, -1])

# TODO: Prettify this
sizes = tf.reduce_sum(tf.cast(tf.reduce_all(tf.not_equal(boxes, -1.), -1), tf.int32), axis=1)

C_split = tf.split(C, sizes, -1)

indices = [linear_sum_assignment(C_split[i][i]) for i in range(len(C_split))]

# TODO: Investigate getting the final losses directly from C using the LSA indices, instead of getting bounding boxes
# from LSA indices and then computing loss.

#%%
# TODO: Get cost matrices in pure tensorflow?
C_splitnp = [split.numpy() for split in C_split]
cost_matrices = tf.ragged.constant(C_splitnp)

lsa = batch_linear_sum_assignment(cost_matrices)
row_idx = sizes_to_batch_indices(sizes)
row_idx = tf.tile(tf.expand_dims(row_idx, -1), [1, 2])
indcs = tf.stack([row_idx, lsa], axis=0)

prediction_idx = tf.transpose(indcs[:, :, 0])
target_idx = tf.transpose(indcs[:, :, 1])

# %% # Get box assignments
target_labels = tf.gather_nd(labels, target_idx)

# get labels of predicted indices
pred_labels = tf.fill([batch_size, n_box], n_class - 1)
pred_labels = tf.tensor_scatter_nd_update(pred_labels, prediction_idx, target_labels)

# get boxes of predicted indices
target_boxes = tf.gather_nd(boxes, target_idx)
pred_boxes = tf.gather_nd(box_out, prediction_idx)

# %% Weighted Cross-entropy
y_pred = class_out
y_true = tf.one_hot(pred_labels, depth=n_class)

real_class_weight = 1
non_class_weight = 0.1
class_weights = tf.concat([tf.repeat(tf.constant(real_class_weight, dtype=tf.float32), n_class - 1),
                           tf.constant([non_class_weight], dtype=tf.float32)],
                          axis=0)

weights = class_weights * y_true
weights = tf.reduce_sum(weights, axis=-1)

loss_ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
loss_ce = loss_ce * weights
loss_ce = tf.reduce_sum(loss_ce) / tf.reduce_sum(weights)  # 0.49932474
print(loss_ce)

# %% Box losses

# L1 loss
n_boxes = tf.cast(tf.shape(target_boxes)[0], tf.float32)
loss_l1_box = tf.reduce_sum(tf.abs(target_boxes - pred_boxes)) / n_boxes  # 0.034271143
print(loss_l1_box)

# GIoU loss
target_boxes = box_cxcywh_to_xyxy(target_boxes)
pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
giou = boxes_giou(pred_boxes, target_boxes)
giou = tf.linalg.diag_part(giou)
loss_giou_box = tf.reduce_mean(1 - giou)  # 0.37486637
print(loss_giou_box)