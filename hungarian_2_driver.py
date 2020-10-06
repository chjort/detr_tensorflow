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
seq = True

y_true, y_pred = load_samples(as_sequence=seq)
# y_true3 = y_true[1:2, :12]
# y_true3 = tf.pad(y_true3, paddings=[[0, 0], [0, 12], [0, 0]], constant_values=-1)
# y_true = tf.concat([y_true, y_true3], axis=0)

# y_pred3 = y_pred[1:2]
# y_pred = tf.concat([y_pred, y_pred3], axis=0)
print(y_true.shape, y_pred.shape)

if seq:
    batch_size = tf.shape(y_pred)[0]
    seq_len = tf.shape(y_pred)[1]
    n_preds = tf.shape(y_pred)[2]
    pred_dim = tf.shape(y_pred)[3]

    y_true = tf.repeat(y_true, seq_len, axis=0)
    y_pred = tf.reshape(y_pred, [-1, n_preds, pred_dim])
    print(y_true.shape, y_pred.shape)
else:
    batch_size = tf.shape(y_pred)[0]
    seq_len = None
    n_preds = tf.shape(y_pred)[1]
    pred_dim = tf.shape(y_pred)[2]

# all seq
batch_mask = hungarian._get_batch_mask(y_true)
cost_matrix = hungarian._compute_cost_matrix(y_true, y_pred, batch_mask)
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

l1 = hungarian.l1_loss(y_true_boxes_lsa, y_pred_boxes_lsa) * hungarian.l1_weight
giou = hungarian.giou_loss(y_true_boxes_lsa, y_pred_boxes_lsa) * hungarian.giou_weight

# TODO: Cross-entropy
if seq:
    batch_size_ce = tf.shape(y_true)[0]
else:
    batch_size_ce = batch_size
n_class = tf.shape(y_pred)[2] - 4
no_class_labels = tf.cast(tf.fill([batch_size_ce, n_preds], n_class - 1), tf.float32)
y_true_labels_lsa = tf.tensor_scatter_nd_update(no_class_labels, y_pred_idx, y_true_labels_lsa)

ce = hungarian.weighted_cross_entropy_loss(y_true_labels_lsa, y_pred_logits) * hungarian.cross_ent_weight
# 0.501876


# %%
sizes = tf.reduce_sum(tf.cast(batch_mask, tf.int32), axis=1)

seq_indices = tf.range(batch_size * seq_len)
seq_indices = tf.transpose(tf.reshape(seq_indices, [batch_size, -1]))

ce = tf.gather(ce, seq_indices)

l1 = tf.RaggedTensor.from_row_lengths(l1, sizes)
l1 = tf.gather(l1, seq_indices)
l1 = l1.merge_dims(1, 2)

giou = tf.RaggedTensor.from_row_lengths(giou, sizes)
giou = tf.gather(giou, seq_indices)
giou = giou.merge_dims(1, 2)

ce
ce_layer = tf.reduce_mean(ce, axis=-1)
l1_layer = tf.reduce_mean(l1, axis=-1)
giou_layer = tf.reduce_mean(giou, axis=-1)

print(ce_layer)
print(l1_layer)
print(giou_layer)

# %%
loss = ce_layer[-1] + l1_layer[-1] + giou_layer[-1]
seq_loss = tf.reduce_sum(ce_layer) + tf.reduce_sum(l1_layer) + tf.reduce_sum(giou_layer)

loss
seq_loss
# tf.Tensor(0.91030025, shape=(), dtype=float32)
# tf.Tensor(5.6704035, shape=(), dtype=float32)

# %%
# sizes = tf.reduce_sum(tf.cast(batch_mask, tf.int32), axis=1)
# sizes
#
# y_true_idx
# y_pred_idx
#
# tf.split(y_true_idx, 2)[0]
#
# l1.shape
# giou.shape
#
# tf.reshape(sizes, [batch_size, -1])

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
