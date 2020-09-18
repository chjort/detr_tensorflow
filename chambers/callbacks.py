from chambers.losses.hungarian import HungarianLoss
import tensorflow as tf
import json


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
        self.hungarian = HungarianLoss(loss_weights=self.model.loss.loss_weights,
                                       lsa_loss_weights=self.model.loss.lsa_loss_weights,
                                       mask_value=self.model.loss.mask_value,
                                       sequence_input=self.model.loss.sequence_input,
                                       sum_losses=False
                                       )

    def on_epoch_end(self, epoch, logs=None):
        pass


    # def on_batch_end(self, epoch, logs=None):
    #     batch_losses = self.model.loss.batch_losses
    #
    #     for i in tf.range(tf.shape(batch_losses)[0]):
    #         losses_i = batch_losses[i]
    #         for loss, name in zip(losses_i, self.loss_names):
    #             if i > 0:
    #                 name = "{}_{}".format(name, i)
    #             logs[name] = loss

class DETR_FB_Loss_Diff(tf.keras.callbacks.Callback):
    def __init__(self):
        super(DETR_FB_Loss_Diff, self).__init__()

        with open("fb_log.txt", "r") as f:
            log_dicts = f.read().split("\n")
            log_dicts = [json.loads(log_dic) for log_dic in log_dicts]

        self.fb_train_loss = [log_dic["train_loss"] for log_dic in log_dicts]
        self.fb_test_loss = [log_dic["test_loss"] for log_dic in log_dicts]

    def on_epoch_end(self, epoch, logs=None):
        train_diff = logs["loss"] - self.fb_train_loss[epoch]
        test_diff = logs["val_loss"] - self.fb_test_loss[epoch]

        logs["train_diff"] = train_diff
        logs["test_diff"] = test_diff
