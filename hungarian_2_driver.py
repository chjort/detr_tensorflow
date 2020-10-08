import pickle

import numpy as np
import tensorflow as tf

from chambers.losses.hungarian_2 import HungarianLoss, batch_linear_sum_assignment_ragged
from chambers.losses.hungarian import HungarianLoss as HL
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


seq = True
hungarian = HungarianLoss(n_classes=91,
                          loss_weights=[1, 5, 2],
                          lsa_loss_weights=[1, 5, 2],
                          mask_value=-1.,
                          sequence_input=seq)
y_true, y_pred = load_samples(as_sequence=seq)

# %%
if hungarian.sequence_input:
    tf.assert_rank(y_pred, 4, "Invalid input shape.")
    batch_size = tf.shape(y_pred)[0]
    seq_len = tf.shape(y_pred)[1]
    n_preds = tf.shape(y_pred)[2]
    pred_dim = tf.shape(y_pred)[3]
    batch_size_flat = batch_size * seq_len

    y_true = tf.repeat(y_true, seq_len, axis=0)
    y_pred = tf.reshape(y_pred, [-1, n_preds, pred_dim])
else:
    batch_size = tf.shape(y_pred)[0]
    seq_len = None
    n_preds = tf.shape(y_pred)[1]
    pred_dim = tf.shape(y_pred)[2]
    batch_size_flat = batch_size

batch_mask = hungarian._get_batch_mask(y_true)
cost_matrix = hungarian._compute_cost_matrix(y_true, y_pred, batch_mask)

lsa = batch_linear_sum_assignment_ragged(cost_matrix)
y_pred_idx, y_true_idx = lsa_to_batch_indices(lsa, batch_mask)

y_true_lsa = tf.gather_nd(y_true, y_true_idx)
y_pred_lsa = tf.gather_nd(y_pred, y_pred_idx)

y_true_boxes_lsa = y_true_lsa[..., :-1]
y_true_labels_lsa = y_true_lsa[..., -1]
y_pred_boxes_lsa = y_pred_lsa[..., :4]
y_pred_logits = y_pred[..., 4:]

n_class = pred_dim - 4
no_class_labels = tf.cast(tf.fill([batch_size_flat, n_preds], n_class - 1), tf.float32)
y_true_labels_lsa = tf.tensor_scatter_nd_update(no_class_labels, y_pred_idx, y_true_labels_lsa)

# compute cross-entropy loss
loss_ce = hungarian.ce_loss_fn(y_true_labels_lsa, y_pred_logits) * hungarian.cross_ent_weight
weights_ce = hungarian._compute_ce_weights(y_true_labels_lsa)
loss_ce = loss_ce * weights_ce

# compute bbox l1 loss
loss_l1 = hungarian.l1_loss_fn(y_true_boxes_lsa, y_pred_boxes_lsa) * hungarian.l1_weight
raw_l1 = loss_l1

# compute bbox giou loss
loss_giou = hungarian.giou_loss_fn(y_true_boxes_lsa, y_pred_boxes_lsa) * hungarian.giou_weight

if hungarian.sequence_input:
    n_trues_per_seq = tf.reduce_sum(tf.cast(batch_mask, tf.int32), axis=1)
    seq_indices = tf.range(batch_size * seq_len)
    seq_indices = tf.transpose(tf.reshape(seq_indices, [batch_size, -1]))

    loss_ce = tf.gather(loss_ce, seq_indices)
    weights_ce = tf.gather(weights_ce, seq_indices)

    # partitions = tf.repeat(
    #     tf.tile(tf.range(seq_len), [batch_size]),
    #     n_trues_per_seq
    # )

    # loss_l1 = tf.dynamic_partition(loss_l1, partitions, seq_len)
    loss_l1 = tf.RaggedTensor.from_row_lengths(loss_l1, n_trues_per_seq)
    loss_l1 = tf.gather(loss_l1, seq_indices)
    loss_l1 = loss_l1.merge_dims(1, 2)

    # loss_giou = tf.dynamic_partition(loss_giou, partitions, seq_len)
    loss_giou = tf.RaggedTensor.from_row_lengths(loss_giou, n_trues_per_seq)
    loss_giou = tf.gather(loss_giou, seq_indices)
    loss_giou = loss_giou.merge_dims(1, 2)

    loss_l1 = tf.reduce_mean(loss_l1, axis=1)
    loss_giou = tf.reduce_mean(loss_giou, axis=1)
else:
    loss_l1 = tf.reduce_mean(loss_l1, axis=0)
    loss_giou = tf.reduce_mean(loss_giou, axis=0)

loss_ce = tf.reduce_sum(loss_ce, axis=(-2, -1)) / tf.reduce_sum(weights_ce, axis=(-2, -1))

losses = [loss_ce, loss_l1, loss_giou]

# %%
data = y_true_idx[:,0] # TODO: find indices for values (0, 6), (1, 7), (2, 8), ..., (5, 11) in `data`
data
# data = raw_l1
data_idx = tf.range(tf.shape(data)[0])
n_trues_per_batch = tf.constant([2, 24])

data_idx
n_trues_per_seq
n_trues_per_batch
n_trues_per_batch * seq_len
seq_indices

tf.stack([data_idx, data], axis=1)
n_trues_per_batch
tf.range(12, 36) + 24*2
batch_mask

tf.constant([0, 1])[:, None] + tf.range(seq_len) * 2
tf.constant([12, 36])[:, None] + tf.range(seq_len) * 24

# index_masks = tf.cast(tf.equal(data, tf.expand_dims(seq_indices, -1)), dtype=tf.int32)
# index_masks = tf.cast(tf.reduce_sum(index_masks, axis=1), tf.bool)
# data = tf.tile(tf.expand_dims(data, 0), [seq_len, 1])
# tf.ragged.boolean_mask(data, index_masks).to_tensor()

# tf.equal(tf.expand_dims(data, 1), tf.expand_dims(tf.range(seq_len*2), 0))

# 156 - tf.range(seq_len) * 24


# %%

# y_true3 = y_true[1:2, :12]
# y_true3 = tf.pad(y_true3, paddings=[[0, 0], [0, 12], [0, 0]], constant_values=-1)
# y_true = tf.concat([y_true, y_true3], axis=0)

# y_pred3 = y_pred[1:2]
# y_pred = tf.concat([y_pred, y_pred3], axis=0)
# print(y_true.shape, y_pred.shape)

# hl = HL(n_classes=91,
#         loss_weights=[1, 5, 2],
#         lsa_loss_weights=[1, 5, 2],
#         mask_value=-1.,
#         sum_losses=False,
#         sequence_input=seq)

# losses_h1 = hl(y_true, y_pred)

# %%
# if seq:
#     input_signature = [tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32),
#                        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)]
# else:
#     input_signature = [tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32),
#                        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)]
#
#
# @tf.function(input_signature=input_signature)
# def compute_loss(y_true, y_pred):
#     if hungarian.sequence_input:
#         tf.assert_rank(y_pred, 4, "Invalid input shape.")
#         batch_size = tf.shape(y_pred)[0]
#         seq_len = tf.shape(y_pred)[1]
#         n_preds = tf.shape(y_pred)[2]
#         pred_dim = tf.shape(y_pred)[3]
#         batch_size_flat = batch_size * seq_len
#
#         y_true = tf.repeat(y_true, seq_len, axis=0)
#         y_pred = tf.reshape(y_pred, [-1, n_preds, pred_dim])
#     else:
#         batch_size = tf.shape(y_pred)[0]
#         seq_len = None
#         n_preds = tf.shape(y_pred)[1]
#         pred_dim = tf.shape(y_pred)[2]
#         batch_size_flat = batch_size
#
#     batch_mask = hungarian._get_batch_mask(y_true)
#     cost_matrix = hungarian._compute_cost_matrix(y_true, y_pred, batch_mask)
#
#     lsa = batch_linear_sum_assignment_ragged(cost_matrix)
#     y_pred_idx, y_true_idx = lsa_to_batch_indices(lsa, batch_mask)
#
#     y_true_lsa = tf.gather_nd(y_true, y_true_idx)
#     y_pred_lsa = tf.gather_nd(y_pred, y_pred_idx)
#
#     y_true_boxes_lsa = y_true_lsa[..., :-1]
#     y_true_labels_lsa = y_true_lsa[..., -1]
#     y_pred_boxes_lsa = y_pred_lsa[..., :4]
#     y_pred_logits = y_pred[..., 4:]
#
#     n_class = pred_dim - 4
#     no_class_labels = tf.cast(tf.fill([batch_size_flat, n_preds], n_class - 1), tf.float32)
#     y_true_labels_lsa = tf.tensor_scatter_nd_update(no_class_labels, y_pred_idx, y_true_labels_lsa)
#
#     # compute cross-entropy loss
#     loss_ce = hungarian.ce_loss_fn(y_true_labels_lsa, y_pred_logits) * hungarian.cross_ent_weight
#     weights_ce = hungarian._compute_ce_weights(y_true_labels_lsa)
#     loss_ce = loss_ce * weights_ce
#
#     # compute bbox l1 loss
#     loss_l1 = hungarian.l1_loss_fn(y_true_boxes_lsa, y_pred_boxes_lsa) * hungarian.l1_weight
#
#     # compute bbox giou loss
#     loss_giou = hungarian.giou_loss_fn(y_true_boxes_lsa, y_pred_boxes_lsa) * hungarian.giou_weight
#
#     if hungarian.sequence_input:
#         sizes = tf.reduce_sum(tf.cast(batch_mask, tf.int32), axis=1)
#         seq_indices = tf.range(batch_size * seq_len)
#         seq_indices = tf.transpose(tf.reshape(seq_indices, [batch_size, -1]))
#
#         loss_ce = tf.gather(loss_ce, seq_indices)
#         weights_ce = tf.gather(weights_ce, seq_indices)
#
#         loss_l1 = tf.RaggedTensor.from_row_lengths(loss_l1, sizes)
#         loss_l1 = tf.gather(loss_l1, seq_indices)
#         loss_l1 = loss_l1.merge_dims(1, 2)
#
#         loss_giou = tf.RaggedTensor.from_row_lengths(loss_giou, sizes)
#         loss_giou = tf.gather(loss_giou, seq_indices)
#         loss_giou = loss_giou.merge_dims(1, 2)
#
#         loss_l1 = tf.reduce_mean(loss_l1, axis=1)
#         loss_giou = tf.reduce_mean(loss_giou, axis=1)
#     else:
#         loss_l1 = tf.reduce_mean(loss_l1, axis=0)
#         loss_giou = tf.reduce_mean(loss_giou, axis=0)
#
#     loss_ce = tf.reduce_sum(loss_ce, axis=(-2, -1)) / tf.reduce_sum(weights_ce, axis=(-2, -1))
#
#     losses = [loss_ce, loss_l1, loss_giou]
#
#     return losses

# %%
# loss = compute_loss(y_true, y_pred)
# tf.stack(loss, axis=1)
# tf.reduce_sum(loss)
# 1.4096249
# 8.68166

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
