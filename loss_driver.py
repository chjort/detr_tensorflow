import pickle

import numpy as np
import tensorflow as tf

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
# TODO: Implement labels loss (Cross-entropy)