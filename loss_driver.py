import pickle

import numpy as np
import tensorflow as tf

from chambers.losses import HungarianLoss
from chambers.utils.tf import batch_linear_sum_assignment

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

# class_out = tf.transpose(tf.stack([*[aux["pred_logits"] for aux in pred["aux_outputs"]], class_out], axis=0),
#                          [1, 0, 2, 3])
# box_out = tf.transpose(tf.stack([*[aux["pred_boxes"] for aux in pred["aux_outputs"]], box_out], axis=0), [1, 0, 2, 3])

boxes = [target["boxes"] for target in targets]
labels = [target["labels"] for target in targets]

boxes1 = tf.pad(boxes[0], paddings=[[0, 22], [0, 0]], constant_values=-1.)
boxes2 = boxes[1]
boxes = tf.stack([boxes1, boxes2], axis=0)

labels1 = tf.pad(labels[0], paddings=[[0, 22]], constant_values=-1)
labels2 = labels[1]
labels = tf.cast(tf.stack([labels1, labels2], axis=0), tf.float32)

y_true = tf.concat([boxes, tf.expand_dims(labels, -1)], axis=-1)
y_pred = tf.concat([box_out, class_out], axis=-1)

y_true.shape
y_pred.shape

# %%
# seq_len = tf.shape(class_out)[1]
# labels = tf.repeat(labels, seq_len, axis=0)  # TODO: Maybe expect shape of input ground truth
# boxes = tf.repeat(boxes, seq_len, axis=0)  # [batch_size, seq_len, n_pred_boxes, 4]
# class_out = tf.reshape(class_out, [-1, tf.shape(class_out)[2], tf.shape(class_out)[3]])
# box_out = tf.reshape(box_out, [-1, tf.shape(box_out)[2], tf.shape(box_out)[3]])

# %%
hungarian = HungarianLoss(mask_value=-1., sequence_input=False)
loss = hungarian(y_true, y_pred)
# loss = hungarian((labels, boxes), (class_out, box_out))
print(loss)  # no seq 1.4204, seq 8.8313

# %%
# Before LSA
y_true_logits = labels  # Ground truth labels   # [batch_size, 24]                          # [2, 24]
y_true_boxes = boxes  # Ground truth boxes    # [batch_size, 24, 4]                       # [2, 24, 4]
y_pred_logits = class_out  # Predicted logits      # [batch_size, n_pred_boxes, n_classes]     # [2, 100, 92]
y_pred_boxes = box_out  # Predicted boxes       # [batch_size, n_pred_boxes, 4]             # [2, 100, 4]
print(y_true_logits.shape)
print(y_true_boxes.shape)
print(y_pred_logits.shape)
print(y_pred_boxes.shape)

batch_size = tf.shape(y_pred_logits)[0]
n_pred_boxes = tf.shape(y_pred_logits)[1]
n_class = tf.shape(y_pred_logits)[2]
batch_mask = hungarian._get_batch_mask(y_true_boxes)
cost_matrix = hungarian._compute_cost_matrix(y_true_logits, y_true_boxes, y_pred_logits, y_pred_boxes, batch_mask)
cost_matrix.bounding_shape()
lsa = batch_linear_sum_assignment(cost_matrix)
prediction_indices, target_indices = hungarian._lsa_to_gather_indices(lsa, batch_mask)
# get assigned targets
y_true_logits_lsa = tf.gather_nd(y_true_logits, target_indices)
y_true_boxes_lsa = tf.gather_nd(y_true_boxes, target_indices)
no_class_labels = tf.cast(tf.fill([batch_size, n_pred_boxes], n_class - 1), tf.float32)
y_true_logits_lsa = tf.tensor_scatter_nd_update(no_class_labels, prediction_indices, y_true_logits_lsa)
# get assigned predictions
y_pred_boxes_lsa = tf.gather_nd(y_pred_boxes, prediction_indices)

# After LSA
print(y_true_logits_lsa.shape)
print(y_true_boxes_lsa.shape)
print(y_pred_logits.shape)
print(y_pred_boxes_lsa.shape)

# Training losses
loss_ce = hungarian.weighted_cross_entropy_loss(y_true_logits_lsa, y_pred_logits) * hungarian.class_loss_weight
loss_l1 = hungarian.l1_loss(y_true_boxes_lsa, y_pred_boxes_lsa) * hungarian.bbox_loss_weight
loss_giou = hungarian.giou_loss(y_true_boxes_lsa, y_pred_boxes_lsa) * hungarian.giou_loss_weight

loss_ce + loss_l1 + loss_giou
