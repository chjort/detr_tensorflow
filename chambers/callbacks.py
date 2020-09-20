import json
import os
from collections.abc import Iterable

import tensorflow as tf

from chambers.losses.hungarian import HungarianLoss as _HungarianLoss
from chambers.models.detr import post_process
from chambers.utils.image import read_image, resnet_imagenet_normalize
from chambers.utils.utils import plot_results
from data.coco import CLASSES


class GroupedTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.log_dir = os.path.join(log_dir, "validation", "decode_layer1")
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            tf.summary.scalar("decode_epoch_loss_ce", epoch, epoch)


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            logs["lr"] = lr(epoch)
        else:
            logs["lr"] = lr


class HungarianLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, dataset: tf.data.Dataset, steps=None):
        super(HungarianLossLogger, self).__init__()
        self.dataset = dataset
        self.steps = steps
        self.loss_names = ["loss_ce", "loss_l1", "loss_giou"]
        self.hungarian = None
        self.y_true = None
        self.batch_size = None

    def on_train_begin(self, logs=None):
        if not isinstance(self.model.loss, _HungarianLoss):
            raise ValueError("Model is not compiled with HungarianLoss. This callback can " \
                             "only be used when the model is compiled with HungarianLoss.")

        if self.hungarian is None:
            self.hungarian = _HungarianLoss(loss_weights=self.model.loss.loss_weights,
                                            lsa_loss_weights=self.model.loss.lsa_loss_weights,
                                            mask_value=self.model.loss.mask_value,
                                            sequence_input=self.model.loss.sequence_input,
                                            sum_losses=False
                                            )
        if self.steps is None:
            self.y_true = [y for x, y in self.dataset]
        else:
            self.y_true = [y for x, y in self.dataset.take(self.steps)]
        self.batch_size = self.y_true[0].shape[0]

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.dataset, steps=self.steps)
        y_pred = tf.split(y_pred, y_pred.shape[0] // self.batch_size)
        losses = [self.hungarian(yt, yp) for yt, yp in zip(self.y_true, y_pred)]
        losses = tf.reduce_mean(losses, axis=0).numpy()

        logs["val_loss"] = losses.sum()

        # the loss of the last decoder layer. (The actual predictions)
        losses_last = losses[-1]
        for loss, name in zip(losses_last, self.loss_names):
            log_name = "val_{}".format(name)
            logs[log_name] = loss

        # the losses of the preceding decoder layers (auxiliary losses).
        for i in range(losses.shape[0] - 1):
            losses_i = losses[i]
            for loss, name in zip(losses_i, self.loss_names):
                log_name = "val_{}_{}".format(name, i)
                logs[log_name] = loss


class DETRLossDiffLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super(DETRLossDiffLogger, self).__init__()
        self.log_file = log_file

        with open(self.log_file, "r") as f:
            log_dicts = f.read().split("\n")
            log_dicts = [json.loads(log_dic) for log_dic in log_dicts]

        self.fb_train_loss = [log_dic["train_loss"] for log_dic in log_dicts]
        self.fb_test_loss = [log_dic["test_loss"] for log_dic in log_dicts]

    def on_epoch_end(self, epoch, logs=None):
        train_diff = logs["loss"] - self.fb_train_loss[epoch]
        test_diff = logs["val_loss"] - self.fb_test_loss[epoch]

        logs["train_diff"] = train_diff
        logs["test_diff"] = test_diff


class DETRPredImageTensorboard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, image_files, min_prob=(0.0, 0.1, 0.5, 0.7)):
        super(DETRPredImageTensorboard, self).__init__()
        self.log_dir = os.path.join(log_dir, "validation")
        self.image_files = image_files
        self.min_prob = list(min_prob) if isinstance(min_prob, Iterable) else [min_prob]
        self.images = []
        self.preprocessed_images = []
        self.writer = None

    def on_train_begin(self, logs=None):
        self.images = [read_image(fp) for fp in self.image_files]
        self.preprocessed_images = [tf.expand_dims(resnet_imagenet_normalize(x), 0) for x in self.images]
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        preds = [self.model(x, training=False) for x in self.preprocessed_images]

        pred_imgs = {}
        for i, (img, pred) in enumerate(zip(self.images, preds)):
            for keep in self.min_prob:
                pred_img = self._draw_predictons(img, pred, CLASSES, min_prob=keep)
                pred_imgs.setdefault("Prediction sample {} - min_probs: {}".format(i, self.min_prob), []).append(
                    pred_img)

        with self.writer.as_default():
            for name, imgs in pred_imgs.items():
                tf.summary.image(name, imgs, max_outputs=len(imgs), step=epoch)

    def _draw_predictons(self, img, y_pred, label_names, min_prob=None, figsize=None, fontsize=None):
        boxes_pred, labels_pred, probs_pred = post_process(y_pred, min_prob=min_prob)
        label_names_pred = [label_names[label] for label in labels_pred]
        pred_img = plot_results(img, boxes_pred, label_names_pred, probs_pred, linewidth=1.5, figsize=figsize,
                                fontsize=fontsize, return_img=True)
        return pred_img
