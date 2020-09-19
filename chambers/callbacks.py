import json

import tensorflow as tf

from chambers.losses.hungarian import HungarianLoss as _HungarianLoss


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            logs["lr"] = lr(epoch)
        else:
            logs["lr"] = lr


class HungarianLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        super(HungarianLossLogger, self).__init__()
        self.dataset = dataset
        self.loss_names = ["loss_ce", "loss_l1", "loss_giou"]
        self.hungarian = None
        self.y_true = [y for x, y in self.dataset]
        self.batch_size = self.y_true[0].shape[0]

    def on_epoch_end(self, epoch, logs=None):
        if self.hungarian is None:
            if not isinstance(self.model.loss, _HungarianLoss):
                raise ValueError("Model is not compiled with HungarianLoss. This callback can " \
                                 "only be used when the model is compiled with HungarianLoss.")
            self.hungarian = _HungarianLoss(loss_weights=self.model.loss.loss_weights,
                                            lsa_loss_weights=self.model.loss.lsa_loss_weights,
                                            mask_value=self.model.loss.mask_value,
                                            sequence_input=self.model.loss.sequence_input,
                                            sum_losses=False
                                            )

        y_pred = self.model.predict(self.dataset)
        y_pred = tf.split(y_pred, y_pred.shape[0] // self.batch_size)
        losses = [self.hungarian(yt, yp) for yt, yp in zip(self.y_true, y_pred)]

        # losses = [self.hungarian(y, self.model(x, training=False)) for x, y in self.dataset]
        losses = tf.reduce_mean(losses, axis=0).numpy()

        logs["val_loss"] = losses.sum()  # NOTE: Values seem to differ from validation data in fit loop.

        for i in range(losses.shape[0]):
            losses_i = losses[i]
            for loss, name in zip(losses_i, self.loss_names):
                log_name = "val_{}_{}".format(name, i)
                logs[log_name] = loss


class DETR_FB_Loss_Diff(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super(DETR_FB_Loss_Diff, self).__init__()
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
