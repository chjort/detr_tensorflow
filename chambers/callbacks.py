import tensorflow as tf


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            logs["lr"] = lr(epoch)
        else:
            logs["lr"] = lr


class HungarianLossLogger(tf.keras.callbacks.Callback):
    def on_batch_end(self, epoch, logs=None):
        # loss_ce = self.model.loss.batch_loss_ce
        # loss_l1 = self.model.loss.batch_loss_l1
        # loss_giou = self.model.loss.batch_loss_giou

        # logs["loss_ce"] = loss_ce
        # logs["loss_l1"] = loss_l1
        # logs["loss_giou"] = loss_giou
        for k, v in self.model.loss.batch_losses.items():
            logs[k] = v
