import json
import os
from collections.abc import Iterable

import tensorflow as tf

from chambers.losses.hungarian import HungarianLoss as _HungarianLoss
from chambers.models.detr import post_process
from chambers.utils.image import read_image, resnet_imagenet_normalize
from chambers.utils.utils import plot_results


class GroupedTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self,
                 loss_groups,
                 writer_prefixes="writer",
                 log_dir='logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 update_freq='epoch',
                 profile_batch=2,
                 embeddings_freq=0,
                 embeddings_metadata=None,
                 **kwargs):
        super(GroupedTensorBoard, self).__init__(log_dir=log_dir,
                                                 histogram_freq=histogram_freq,
                                                 write_graph=write_graph,
                                                 write_images=write_images,
                                                 update_freq=update_freq,
                                                 profile_batch=profile_batch,
                                                 embeddings_freq=embeddings_freq,
                                                 embeddings_metadata=embeddings_metadata,
                                                 **kwargs)
        self.loss_groups = loss_groups
        self.writer_prefixes = writer_prefixes
        self.log_dir_groups = os.path.join(self.log_dir, "groups")
        self.writers = None

    def on_train_begin(self, logs=None):
        os.makedirs(self.log_dir_groups, exist_ok=True)
        super(GroupedTensorBoard, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        if not isinstance(self.loss_groups[0], str):
            # if nested list

            if isinstance(self.writer_prefixes, str) or len(self.writer_prefixes) != len(self.loss_groups):
                raise ValueError("`writer_prefixes` must be a list of strings with same length as `loss_groups`.")

            losses = {}
            for loss_group, writer_prefix in zip(self.loss_groups, self.writer_prefixes):
                writer_losses, logs = self._filter_writer_losses(logs, loss_group, writer_prefix)
                losses.update(writer_losses)
        else:
            if not isinstance(self.writer_prefixes, str):
                raise ValueError("`writer_prefixes` must be a string when there is only one loss group.")
            losses, logs = self._filter_writer_losses(logs, self.loss_groups, self.writer_prefixes)

        if self.writers is None:
            # create writers
            self.writers = {k: tf.summary.create_file_writer(os.path.join(self.log_dir_groups, k)) for k in
                            losses.keys()}

        for writer_name, loss_dict in losses.items():
            writer = self.writers[writer_name]
            with writer.as_default():
                for loss_name, loss_val in loss_dict.items():
                    tf.summary.scalar(loss_name, data=loss_val, step=epoch)

        # Call super TensorBoard
        super(GroupedTensorBoard, self).on_epoch_end(epoch, logs)

    @staticmethod
    def _filter_writer_losses(logs, loss_names, writer_prefix):
        """
        Gets losses for each writer

        Example:
            >>> logs = {
                    'loss': 0.34, 'val_loss': 0.41
                    'A': 1, 'B': 2, 'C': 3,
                    'A_0': 4, 'B_0': 5, 'C_0': 6,
                    'A_1': 7, 'B_1': 8, 'C_1': 9
                }
            >>> loss_names = ['A', 'B', 'C']
            >>> writer_prefix = "writer"
            >>> self._filter_writer_losses(logs, loss_names, writer_prefix)
            {
                'writer' : {'A': 1, 'B': 2, 'C': 3},
                'writer_0' {'A': 4, 'B': 5, 'C': 6},
                'writer_1' {'A': 7, 'B': 8, 'C': 9},
            },
            {'loss': 0.34, 'val_loss': 0.41}

        :param logs: Dictionary with keys
        :param loss_names:
        :return: dictionary with writer names and losses, and a logs dictionary where found losses are removed.
        """
        logs = logs.copy()
        losses = {}
        for loss_name in loss_names:
            try:
                loss_val = logs.pop(loss_name)
                losses.setdefault(writer_prefix, {}).update({loss_name: loss_val})
            except KeyError:
                pass

            do = True
            i = 0
            while do:
                try:
                    k = loss_name + "_" + str(i)
                    loss_val = logs.pop(k)
                    writer_name = writer_prefix + "_" + str(i)
                    losses.setdefault(writer_name, {}).update({loss_name: loss_val})
                    i = i + 1
                except KeyError:
                    do = False
        return losses, logs


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
        self._supports_tf_logs = True
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
            # TODO: Switch to new optimized HungarianLoss
            self.hungarian = _HungarianLoss(n_classes=self.model.loss.n_classes,
                                            no_class_weight=self.model.loss.no_class_weight,
                                            loss_weights=self.model.loss.loss_weights,
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

        # TODO: Handle new optimized loss
        total_losses = [tf.reduce_sum(loss) for loss in losses]
        val_loss = tf.reduce_mean(total_losses)
        logs["val_loss"] = val_loss.numpy()

        losses = tf.reduce_mean(losses, axis=0).numpy()

        # the loss of the last decoder layer. (The actual predictions)
        losses_last = losses[:, -1]
        for loss, name in zip(losses_last, self.loss_names):
            log_name = "val_{}".format(name)
            logs[log_name] = loss

        # the losses of the preceding decoder layers (auxiliary losses).
        for i in range(losses.shape[1] - 1):
            losses_i = losses[:, i]
            for loss, name in zip(losses_i, self.loss_names):
                log_name = "val_{}_{}".format(name, i)
                logs[log_name] = loss


class DETRLossDiffLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super(DETRLossDiffLogger, self).__init__()
        self._supports_tf_logs = True
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
    def __init__(self, log_dir, image_files, min_prob=(0.0, 0.1, 0.5, 0.7), label_names=None):
        super(DETRPredImageTensorboard, self).__init__()
        self.log_dir = os.path.join(log_dir, "validation")
        self.image_files = image_files
        self.min_prob = list(min_prob) if isinstance(min_prob, Iterable) else [min_prob]
        self.label_names = label_names
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
                pred_img = self._draw_predictons(img, pred, min_prob=keep, fontsize=20)
                pred_imgs.setdefault("Prediction sample {} - min_probs: {}".format(i, self.min_prob), []).append(
                    pred_img)

        with self.writer.as_default():
            for name, imgs in pred_imgs.items():
                tf.summary.image(name, imgs, max_outputs=len(imgs), step=epoch)

    def _draw_predictons(self, img, y_pred, min_prob=None, figsize=None, fontsize=None):
        boxes_pred, labels_pred, probs_pred = post_process(y_pred, min_prob=min_prob)

        if self.label_names is not None:
            label_names_pred = [self.label_names[label] for label in labels_pred]
        else:
            label_names_pred = labels_pred

        pred_img = plot_results(img, boxes_pred, label_names_pred, probs_pred, linewidth=1.5, figsize=figsize,
                                fontsize=fontsize, return_img=True)
        return pred_img
