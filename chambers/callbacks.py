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
        batch_losses = self.model.loss.batch_losses
        loss_names = self.model.loss.loss_names

        for i in tf.range(tf.shape(batch_losses)[0]):
            losses_i = batch_losses[i]
            for loss, name in zip(losses_i, loss_names):
                if i > 0:
                    name = "{}_{}".format(name, i)
                logs[name] = loss
