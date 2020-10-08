import pickle
import time

import numpy as np
import tensorflow as tf

from chambers.losses.hungarian import HungarianLoss
from chambers.utils.boxes import box_cxcywh_to_yxyx


def load_samples(as_sequence=False):
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
y_true, y_pred = load_samples()
loss = hungarian(y_true, y_pred)

hungarian(y_true, y_pred)  # warmup
times = []
for i in range(200):
    st = time.time()
    hungarian(y_true, y_pred)
    end_time = time.time() - st
    times.append(end_time)
print(np.mean(times))  # 0.006532610654830933

# with sequence
hungarian = HungarianLoss(n_classes=91,
                          loss_weights=[1, 5, 2],
                          lsa_loss_weights=[1, 5, 2],
                          mask_value=-1.,
                          sequence_input=True)
y_true, y_pred = load_samples(as_sequence=True)
seq_loss = hungarian(y_true, y_pred)

hungarian(y_true, y_pred)  # warmup
times = []
for i in range(200):
    st = time.time()
    hungarian(y_true, y_pred)
    end_time = time.time() - st
    times.append(end_time)
print(np.mean(times))  # 0.020411419868469238

print(loss)
print(seq_loss)
tf.assert_equal(loss, 1.409625)
tf.assert_equal(seq_loss, 8.68166)
