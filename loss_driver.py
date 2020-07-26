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

# %%
# TODO: Implement labels loss (Cross-entropy)


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
indices = [(tf.convert_to_tensor(i, dtype=tf.int64), tf.convert_to_tensor(j, dtype=tf.int64)) for i, j in indices]
indices


@tf.function
def tf_linear_sum_assignment(cost_matrix):
    return tf.numpy_function(func=linear_sum_assignment, inp=[cost_matrix], Tout=[tf.int64, tf.int64])


indices_tf = [tf_linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
indices_tf
