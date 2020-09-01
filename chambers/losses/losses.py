import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


class L1Loss(LossFunctionWrapper):
    def __init__(
            self,
            reduction: str = tf.keras.losses.Reduction.AUTO,
            name: str = "l1_loss",
    ):
        super().__init__(l1_loss, name=name, reduction=reduction)


def l1_loss(y_true, y_pred):
    l1_dist = tf.abs(y_true - y_pred)
    # l1_dist = tf.reduce_sum(l1_dist, axis=-1)
    # loss_l1 = tf.reduce_mean(l1_dist)

    n_target_boxes = tf.cast(tf.shape(y_true)[0], tf.float32)
    loss_l1 = tf.reduce_sum(l1_dist) / n_target_boxes
    return loss_l1
