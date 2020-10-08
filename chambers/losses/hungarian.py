import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from chambers.utils.tf import batch_linear_sum_assignment_ragged, lsa_to_batch_indices
from .losses import L1Loss
from .pairwise import pairwise_giou as _pairwise_giou, pairwise_l1 as _pairwise_l1


class HungarianLoss(tf.keras.losses.Loss):
    """
    Computes the Linear Sum Assignment (LSA) between predictions and targets, using a weighted sum of L1,
    Generalized IOU and softmax probabilities for the cost matrix.

    The final loss is given as a weighted sum of cross-entropy, L1 and Generalized IOU on the prediction-target pairs
    given by the LSA.

    """

    def __init__(self, n_classes, loss_weights=None, lsa_loss_weights=None, no_class_weight=0.1, mask_value=None,
                 sequence_input=False, sum_losses=True, name="hungarian_loss", **kwargs):
        self.n_classes = n_classes
        self.no_class_weight = no_class_weight
        self.mask_value = mask_value
        self.sequence_input = sequence_input
        self.sum_losses = sum_losses

        self.ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                        reduction=tf.keras.losses.Reduction.NONE)
        self.l1_loss_fn = L1Loss(reduction=tf.keras.losses.Reduction.NONE)
        self.giou_loss_fn = tfa.losses.GIoULoss("giou", reduction=tf.keras.losses.Reduction.NONE)
        self.lsa_losses = [pairwise_sparse_softmax, pairwise_l1, pairwise_giou]

        self.class_weights = tf.concat([tf.ones(self.n_classes, dtype=tf.float32),
                                        tf.constant([self.no_class_weight], dtype=tf.float32)],
                                       axis=0)

        self.loss_weights = loss_weights
        if self.loss_weights is None:
            self.cross_ent_weight, self.l1_weight, self.giou_weight = 1, 1, 1
        else:
            self.cross_ent_weight, self.l1_weight, self.giou_weight = 1, 5, 2

        self.lsa_loss_weights = lsa_loss_weights
        if self.lsa_loss_weights is None:
            self.lsa_loss_weights = [1, 1, 1]

        if sequence_input:
            self.input_signature = [tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32),
                                    tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)]
        else:
            self.input_signature = [tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32),
                                    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)]
        self.call = tf.function(self.call, input_signature=self.input_signature)

        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)

    def call(self, y_true, y_pred):
        """
        :param y_true: shape [batch_size, n_true_boxes, 5].
        :param y_pred: shape [batch_size, n_pred_boxes, 4 + n_classes].
            If ``self.sequence_input`` is True shape must be [batch_size, sequence_len, n_true_boxes, 5]
        :return:
        """
        batch_mask = self._get_batch_mask(y_true)
        n_true = tf.reduce_sum(tf.cast(batch_mask, tf.float32))

        if self.sequence_input:
            tf.assert_rank(y_pred, 4, "Invalid input shape.")
            batch_size = tf.shape(y_pred)[0]
            seq_len = tf.shape(y_pred)[1]
            n_pred_boxes = tf.shape(y_pred)[2]
            pred_dim = tf.shape(y_pred)[3]
            batch_size_flat = batch_size * seq_len

            y_true = tf.repeat(y_true, seq_len, axis=0)
            y_pred = tf.reshape(y_pred, [-1, n_pred_boxes, pred_dim])
        else:
            batch_size = tf.shape(y_pred)[0]
            seq_len = None
            n_pred_boxes = tf.shape(y_pred)[1]
            pred_dim = tf.shape(y_pred)[2]
            batch_size_flat = batch_size

        batch_mask = self._get_batch_mask(y_true)
        cost_matrix = self._compute_cost_matrix(y_true, y_pred, batch_mask)

        # handle nan in cost matrix (nan can occur from the softmax function used in one of the pairwise losses)
        nan_matrices = tf.reduce_all(tf.math.is_nan(cost_matrix), axis=(1, 2))
        if tf.reduce_all(nan_matrices):
            # if all cost matrices are nan, return nan.
            if self.sum_losses:
                return np.nan
            else:
                return [np.nan, np.nan, np.nan]
        if tf.reduce_any(nan_matrices):
            # remove any matrices that are nan.
            cost_matrix, batch_mask = self._remove_nan_cost_matrix(nan_matrices, cost_matrix, batch_mask)

        lsa = batch_linear_sum_assignment_ragged(cost_matrix)
        y_pred_idx, y_true_idx = lsa_to_batch_indices(lsa, batch_mask)

        y_true_lsa = tf.gather_nd(y_true, y_true_idx)
        y_pred_lsa = tf.gather_nd(y_pred, y_pred_idx)

        y_true_boxes_lsa = y_true_lsa[..., :-1]
        y_true_labels_lsa = y_true_lsa[..., -1]
        y_pred_boxes_lsa = y_pred_lsa[..., :4]
        y_pred_logits = y_pred[..., 4:]

        n_class = pred_dim - 4
        no_class_labels = tf.cast(tf.fill([batch_size_flat, n_pred_boxes], n_class - 1), tf.float32)
        y_true_labels_lsa = tf.tensor_scatter_nd_update(no_class_labels, y_pred_idx, y_true_labels_lsa)

        # compute cross-entropy loss
        loss_ce = self.ce_loss_fn(y_true_labels_lsa, y_pred_logits) * self.cross_ent_weight
        weights_ce = self._compute_ce_weights(y_true_labels_lsa)
        loss_ce = loss_ce * weights_ce

        # compute bbox l1 loss
        loss_l1 = self.l1_loss_fn(y_true_boxes_lsa, y_pred_boxes_lsa) * self.l1_weight

        # compute bbox giou loss
        loss_giou = self.giou_loss_fn(y_true_boxes_lsa, y_pred_boxes_lsa) * self.giou_weight

        if self.sequence_input:
            n_trues_per_seq = tf.reduce_sum(tf.cast(batch_mask, tf.int32), axis=1)
            seq_indices = tf.range(batch_size * seq_len)
            seq_indices = tf.transpose(tf.reshape(seq_indices, [batch_size, -1]))

            loss_ce = tf.gather(loss_ce, seq_indices)
            weights_ce = tf.gather(weights_ce, seq_indices)

            seq_range = tf.range(seq_len)
            partitions = tf.repeat(
                tf.tile(seq_range, [batch_size]),
                n_trues_per_seq
            )
            partition_mask = tf.equal(
                tf.expand_dims(partitions, -1),
                tf.expand_dims(seq_range, 0)
            )
            partition_mask = tf.cast(partition_mask, tf.float32)

            loss_l1 = tf.expand_dims(loss_l1, -1)
            loss_giou = tf.expand_dims(loss_giou, -1)

            loss_l1 = loss_l1 * partition_mask
            loss_giou = loss_giou * partition_mask

            loss_l1 = tf.reduce_sum(loss_l1, axis=0) / n_true
            loss_giou = tf.reduce_sum(loss_giou, axis=0) / n_true
        else:
            loss_l1 = tf.reduce_mean(loss_l1, axis=0)
            loss_giou = tf.reduce_mean(loss_giou, axis=0)

        loss_ce = tf.reduce_sum(loss_ce, axis=(-2, -1)) / tf.reduce_sum(weights_ce, axis=(-2, -1))

        losses = [loss_ce, loss_l1, loss_giou]

        if self.sum_losses:
            losses = tf.reduce_sum(losses)

        return losses

    def _compute_ce_weights(self, y_true):
        y_true_1hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.n_classes + 1)  # +1 for "no object" class
        weights = self.class_weights * y_true_1hot  # multiply weights with target class distribution
        weights = tf.reduce_sum(weights, axis=-1)
        return weights

    def _compute_cost_matrix(self, y_true, y_pred, batch_mask):
        cost_losses = [loss_fn(y_true, y_pred) * loss_weight for loss_fn, loss_weight in
                       zip(self.lsa_losses, self.lsa_loss_weights)]

        cost_matrix = tf.math.add_n(cost_losses)
        cost_matrix_mask = self._compute_cost_matrix_mask(cost_matrix, batch_mask)
        cost_matrix_ragged = tf.ragged.boolean_mask(cost_matrix, cost_matrix_mask)

        return cost_matrix_ragged

    @staticmethod
    def _compute_cost_matrix_mask(cost_matrix, batch_mask):
        n_pred_boxes = tf.shape(cost_matrix)[1]
        cost_matrix_mask = tf.tile(tf.expand_dims(batch_mask, 1), [1, n_pred_boxes, 1])
        return cost_matrix_mask

    @staticmethod
    def _remove_nan_cost_matrix(nan_matrices, cost_matrix, batch_mask):
        no_nan_matrices = tf.logical_not(nan_matrices)
        cost_matrix = tf.ragged.boolean_mask(cost_matrix, no_nan_matrices)
        batch_mask = tf.cast(
            tf.transpose(tf.transpose(tf.cast(batch_mask, tf.float32)) * tf.cast(no_nan_matrices, tf.float32)),
            tf.bool)
        return cost_matrix, batch_mask

    def _get_batch_mask(self, y_true):
        """
        :param y_true:
        :return: [batch_size, max_n_true]
        :rtype:
        """
        return tf.reduce_all(tf.not_equal(y_true, self.mask_value), -1)

    def get_config(self):
        config = {"n_classes": self.n_classes, "loss_weights": self.loss_weights,
                  "lsa_loss_weights": self.lsa_loss_weights, "no_class_weight": self.no_class_weight,
                  "mask_value": self.mask_value, "sequence_input": self.sequence_input, "sum_losses": self.sum_losses}
        base_config = super(HungarianLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def pairwise_giou(y_true, y_pred):
    """
    :param y_true: (batch_size, n_true_boxes, 5)
    :param y_pred: (batch_size, n_pred_boxes, 4 + n_classes)
    :return: (batch_size, n_pred_boxes, n_true_boxes)
    """

    y_true = y_true[..., :-1]  # [batch_size, n_true_boxes, 4]
    y_pred = y_pred[..., :4]  # [batch_size, n_pred_boxes, 4]

    cost_giou = _pairwise_giou(y_true, y_pred)
    return cost_giou


def pairwise_l1(y_true, y_pred):
    """
    :param y_true: (batch_size, n_true_boxes, 5)
    :param y_pred: (batch_size, n_pred_boxes, 4 + n_classes)
    :return: (batch_size, n_pred_boxes, n_true_boxes)
    """

    y_true = y_true[..., :-1]  # [batch_size, n_true_boxes, 4]
    y_pred = y_pred[..., :4]  # [batch_size, n_pred_boxes, 4]

    # bbox cost
    cost_bbox = _pairwise_l1(y_pred, y_true)
    return cost_bbox


def pairwise_sparse_softmax(y_true, y_pred):
    """
    :param y_true: (batch_size, n_true_boxes, 5)
    :param y_pred: (batch_size, n_pred_boxes, 4 + n_classes)
    :return: (batch_size, n_pred_boxes, n_true_boxes)
    """

    y_true = y_true[..., -1]
    y_pred = y_pred[..., 4:]

    # tf.gather does not take indices that are out of bounds when using CPU. So converting out of bounds indices to 0.
    y_true = tf.maximum(y_true, 0)
    y_true = tf.cast(y_true, tf.int32)

    def gather(inputs):
        y_pred, y_true = inputs
        return tf.gather(y_pred, y_true, axis=-1)

    y_pred = tf.nn.softmax(y_pred, axis=-1)
    cost_softmax = -tf.vectorized_map(gather, (y_pred, y_true))
    return cost_softmax


tf.keras.utils.get_custom_objects().update({
    "HungarianLoss": HungarianLoss,
})
