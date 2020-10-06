import pickle

import numpy as np
import tensorflow as tf

from chambers.losses.hungarian_2 import HungarianLoss, batch_linear_sum_assignment_ragged
from chambers.utils.boxes import box_cxcywh_to_yxyx
from chambers.utils.tf import lsa_to_batch_indices

def load_samples(as_sequence=False):
    samples = np.load("test/x_pt.npy")
    samples = tf.convert_to_tensor(samples)
    samples = tf.transpose(samples, [0, 2, 3, 1])

    mask = np.load("test/mask_pt.npy")
    mask = tf.convert_to_tensor(mask)

    x1 = tf.where(tf.expand_dims(mask[0], -1), tf.ones_like(samples[0]) * -1., samples[0])  # masked values to -1
    x2 = tf.where(tf.expand_dims(mask[1], -1), tf.ones_like(samples[1]) * -1., samples[1])  # masked values to -1
    x = tf.stack([x1, x2], axis=0)

    with open("test/y_pt.pickle", "rb") as f:
        targets = pickle.load(f)
        targets = [{k: tf.convert_to_tensor(v) for k, v in t.items() if k in ("boxes", "labels")} for t in targets]

    with open("test/pred_pt.pickle", "rb") as f:
        pred = pickle.load(f)
        pred["pred_logits"] = tf.convert_to_tensor(pred["pred_logits"])
        pred["pred_boxes"] = tf.convert_to_tensor(pred["pred_boxes"])
        pred["aux_outputs"] = [{k: tf.convert_to_tensor(v) for k, v in aux_output.items()}
                               for aux_output in pred["aux_outputs"]]
        class_out = pred["pred_logits"]
        box_out = pred["pred_boxes"]

    if as_sequence:
        class_out = tf.transpose(tf.stack([*[aux["pred_logits"] for aux in pred["aux_outputs"]], class_out], axis=0),
                                 [1, 0, 2, 3])
        box_out = tf.transpose(tf.stack([*[aux["pred_boxes"] for aux in pred["aux_outputs"]], box_out], axis=0),
                               [1, 0, 2, 3])

    boxes = [target["boxes"] for target in targets]
    labels = [target["labels"] for target in targets]

    boxes1 = tf.pad(boxes[0], paddings=[[0, 22], [0, 0]], constant_values=-1.)
    boxes2 = boxes[1]
    boxes = tf.stack([boxes1, boxes2], axis=0)

    labels1 = tf.pad(labels[0], paddings=[[0, 22]], constant_values=-1)
    labels2 = labels[1]
    labels = tf.cast(tf.stack([labels1, labels2], axis=0), tf.float32)

    boxes = box_cxcywh_to_yxyx(boxes)
    box_out = box_cxcywh_to_yxyx(box_out)

    y_true = tf.concat([boxes, tf.expand_dims(labels, -1)], axis=-1)
    y_pred = tf.concat([box_out, class_out], axis=-1)
    return y_true, y_pred


hungarian = HungarianLoss(n_classes=91,
                          loss_weights=[1, 5, 2],
                          lsa_loss_weights=[1, 5, 2],
                          mask_value=-1.,
                          sequence_input=False)
# y_true, y_pred = load_samples()
# loss = hungarian(y_true, y_pred)

# %%
y_true, y_pred = load_samples(as_sequence=False)
print(y_true.shape, y_pred.shape)
sy_true, sy_pred = load_samples(as_sequence=True)
print(sy_true.shape, sy_pred.shape)

batch_size = tf.shape(sy_pred)[0]
seq_len = tf.shape(sy_pred)[1]
n_preds = tf.shape(sy_pred)[2]
pred_dim = tf.shape(sy_pred)[3]

sy_true_r = tf.repeat(sy_true, seq_len, axis=0)
sy_pred_r = tf.reshape(sy_pred, [-1, n_preds, pred_dim])
print(sy_true_r.shape, sy_pred_r.shape)

# all seq
batch_mask = hungarian._get_batch_mask(sy_true_r)
cost_matrix = hungarian._compute_cost_matrix(sy_true_r, sy_pred_r, batch_mask)
print("C shape:", cost_matrix.bounding_shape())

lsa = batch_linear_sum_assignment_ragged(cost_matrix)
print(lsa.shape)

y_pred_idx, y_true_idx = lsa_to_batch_indices(lsa, batch_mask)
print(y_true_idx.shape, y_pred_idx.shape)

###
y_true_lsa = tf.gather_nd(y_true, y_true_idx)
y_pred_lsa = tf.gather_nd(y_pred, y_pred_idx)
print(y_true_lsa.shape, y_pred_lsa.shape)

y_true_boxes_lsa = y_true_lsa[..., :-1]
y_true_labels_lsa = y_true_lsa[..., -1]
y_pred_boxes_lsa = y_pred_lsa[..., :4]
y_pred_logits = y_pred[..., 4:]

l1 = hungarian.l1_loss(y_true_boxes_lsa, y_pred_boxes_lsa)# * hungarian.l1_weight
giou = hungarian.giou_loss(y_true_boxes_lsa, y_pred_boxes_lsa)# * hungarian.giou_weight
l1.shape
giou.shape

#%% single seq
# tf.assert_equal(sy_pred_r[:1], sy_pred[:1, 0, :, :])
i = 11
p_s = sy_pred_r[i:i+1]
y_s = sy_true_r[i:i+1]
sbatch_mask = hungarian._get_batch_mask(y_s)
scost_matrix = hungarian._compute_cost_matrix(y_s, p_s, sbatch_mask)
scost_matrix.bounding_shape()
print("C shape:", scost_matrix.bounding_shape())

tf.assert_equal(cost_matrix[i:i+1].to_tensor(), scost_matrix.to_tensor())

# %% with sequence
# hungarian = HungarianLoss(n_classes=91,
#                           loss_weights=[1, 5, 2],
#                           lsa_loss_weights=[1, 5, 2],
#                           mask_value=-1.,
#                           sequence_input=True)
# y_true, y_pred = load_samples(as_sequence=True)
# seq_loss = hungarian(y_true, y_pred)

# %%
# print(loss)
# print(seq_loss)
# tf.assert_equal(loss, 1.4096249)  # YXYX
# tf.assert_equal(seq_loss, 8.68166)  # YXYX
