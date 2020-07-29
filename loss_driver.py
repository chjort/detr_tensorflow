import pickle

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from chambers.utils.boxes import box_cxcywh_to_xyxy, boxes_giou
from chambers.utils.tf import pairwise_l1

COCO_PATH = "/home/crr/datasets/coco"
BATCH_SIZE = 2

# %%
samples = np.load("x_pt.npy")
samples = tf.convert_to_tensor(samples)
samples = tf.transpose(samples, [0, 2, 3, 1])

mask = np.load("mask_pt.npy")
mask = tf.convert_to_tensor(mask)

with open("y_pt.pickle", "rb") as f:
    targets = pickle.load(f)
    targets = [{k: tf.convert_to_tensor(v) for k, v in t.items() if k in ("boxes", "labels")} for t in targets]

with open("pred_pt.pickle", "rb") as f:
    pred = pickle.load(f)
    pred["pred_logits"] = tf.convert_to_tensor(pred["pred_logits"])
    pred["pred_boxes"] = tf.convert_to_tensor(pred["pred_boxes"])
    pred["aux_outputs"] = [{k: tf.convert_to_tensor(v) for k, v in aux_output.items()}
                           for aux_output in pred["aux_outputs"]]

# %%
outputs = pred
outputs["pred_logits"].shape  # shape (batch_size, n_boxes, n_classes)
outputs["pred_boxes"].shape  # shape (batch_size, n_boxes, 4), box coordinates (center_x, center_y, w, h)
samples.shape
targets[0]

# %% Matcher
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2

# including batch dimension (original output shape)
# single example
# y_pred = outputs["pred_boxes"][1:2]
# y_true = tf.expand_dims(targets[1]["boxes"], 0)

# stacking batches
batch_size, n_box, n_class = outputs["pred_logits"].shape
box_shape = outputs["pred_boxes"].shape[-1]

y_pred_logits = tf.reshape(outputs["pred_logits"], [-1, n_class])  # out_bbox
y_pred_logits = tf.nn.softmax(y_pred_logits, axis=-1)
y_true_logits = tf.concat([target["labels"] for target in targets], axis=0)  # tgt_pred

y_pred_box = tf.reshape(outputs["pred_boxes"], [-1, box_shape])  # out_bbox
y_true_box = tf.concat([target["boxes"] for target in targets], axis=0)  # tgt_pred

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

sizes = [len(v["boxes"]) for v in targets]
C_split = tf.split(C, sizes, -1)

indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
indices = [(tf.convert_to_tensor(i, dtype=tf.int32), tf.convert_to_tensor(j, dtype=tf.int32)) for i, j in indices]

#%%
y_pred_box.shape
y_true_box.shape

C.shape
C_split[0].shape
C_split[1].shape
[c[i].shape for i, c in enumerate(C_split)]

cb0 = C_split[0][0]
cb1 = C_split[1][1]

cb0.shape
cb1.shape
linear_sum_assignment(cb0)
linear_sum_assignment(cb1)

# %% # Get box assignments
row_idx = tf.concat([tf.fill(q_idx.shape, i) for i, (q_idx, k_idx) in enumerate(indices)], axis=0)
col_idx = tf.concat([q_idx for (q_idx, k_idx) in indices], axis=0)
prediction_idx = tf.stack([row_idx, col_idx], axis=1)

# get labels of predicted indices
target_labels = [tf.gather(target["labels"], tgt_idx) for target, (prd_idx, tgt_idx) in zip(targets, indices)]
target_labels = tf.concat(target_labels, axis=0)
target_labels = tf.cast(target_labels, tf.int32)
pred_labels = tf.fill([batch_size, n_box], n_class - 1)
pred_labels = tf.tensor_scatter_nd_update(pred_labels, prediction_idx, target_labels)

# get boxes of predicted indices
target_boxes = tf.concat([tf.gather(target["boxes"], tgt_idx) for target, (prd_idx, tgt_idx) in zip(targets, indices)], axis=0)
pred_boxes = tf.gather_nd(outputs["pred_boxes"], prediction_idx)

# %% Weighted Cross-entropy
y_pred = outputs["pred_logits"]
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
loss_ce = tf.reduce_sum(loss_ce) / tf.reduce_sum(weights)

# %% Box losses
pred_boxes.shape
target_boxes.shape

# L1 loss
n_boxes = tf.cast(tf.shape(target_boxes)[0], tf.float32)
loss_l1_box = tf.reduce_sum(tf.abs(target_boxes - pred_boxes)) / n_boxes

# GIoU loss
target_boxes = box_cxcywh_to_xyxy(target_boxes)
pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
giou = boxes_giou(pred_boxes, target_boxes)
giou = tf.linalg.diag_part(giou)
loss_giou_box = tf.reduce_mean(1 - giou)
