import numpy as np
import tensorflow as tf

from chambers.utils.tf import batch_linear_sum_assignment, repeat_indices
from .iou import GIoULoss
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
        self.weighted_cross_entropy_loss = WeightedSparseCategoricalCrossEntropyCocoDETR(
            n_classes=n_classes, no_class_weight=no_class_weight, reduction=tf.keras.losses.Reduction.NONE)
        self.l1_loss = L1Loss(reduction=tf.keras.losses.Reduction.NONE)
        self.giou_loss = GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
        self.lsa_losses = [pairwise_sparse_softmax, pairwise_l1, pairwise_giou]

        self.loss_weights = loss_weights
        if self.loss_weights is None:
            self.cross_ent_weight, self.l1_weight, self.giou_weight = 1, 1, 1
        else:
            self.cross_ent_weight, self.l1_weight, self.giou_weight = 1, 5, 2

        self.lsa_loss_weights = lsa_loss_weights
        if self.lsa_loss_weights is None:
            self.lsa_loss_weights = [1, 1, 1]

        self.n_classes = n_classes
        self.no_class_weight = no_class_weight
        self.mask_value = mask_value
        self.sequence_input = sequence_input
        self.sum_losses = sum_losses

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
        if self.sequence_input:
            # TODO: Parallelize loss computation over sequence
            tf.assert_rank(y_pred, 4, "Invalid input shape.")
            seq_len = tf.shape(y_pred)[1]

            decode_layers_losses = tf.TensorArray(tf.float32, size=seq_len, element_shape=tf.TensorShape([3]))
            for i in tf.range(seq_len):
                y_pred_i = y_pred[:, i, :, :]
                losses_i = self._compute_losses(y_true, y_pred_i)
                decode_layers_losses = decode_layers_losses.write(i, losses_i)

            losses = decode_layers_losses.stack()
        else:
            losses = [self._compute_losses(y_true, y_pred)]

        if self.sum_losses:
            losses = tf.reduce_sum(losses)

        return losses

    def _compute_losses(self, y_true, y_pred):
        """
        :param y_true: [batch_size, n_true_boxes, 5]
        :param y_pred: [batch_size, n_pred_boxes, 4 + n_classes]
        :return:
        """

        batch_mask = self._get_batch_mask(y_true)  # [batch_size, max_n_true_boxes_batch]

        # cost_matrix (RaggedTensor) [batch_size, n_pred_boxes, None]
        cost_matrix = self._compute_cost_matrix(y_true, y_pred, batch_mask)  # TODO: input format

        # Ignore cost matrices that are all NaN.
        nan_matrices = tf.reduce_all(tf.math.is_nan(cost_matrix), axis=(1, 2))
        if tf.reduce_all(nan_matrices):
            tf.print("Warning: Hungarian NaN:", nan_matrices)
            return np.nan, np.nan, np.nan

        if tf.reduce_any(nan_matrices):
            tf.print("Warning: Hungarian NaN:", nan_matrices)
            no_nan_matrices = tf.logical_not(nan_matrices)
            cost_matrix = tf.ragged.boolean_mask(cost_matrix, no_nan_matrices)
            batch_mask = tf.cast(
                tf.transpose(tf.transpose(tf.cast(batch_mask, tf.float32)) * tf.cast(no_nan_matrices, tf.float32)),
                tf.bool)

        lsa = batch_linear_sum_assignment(cost_matrix)  # [n_true_boxes, 2]

        prediction_indices, target_indices = self._lsa_to_batch_indices(lsa, batch_mask)
        # ([n_true_boxes, 2], [n_true_boxes, 2])

        y_true_lsa = tf.gather_nd(y_true, target_indices)
        y_pred_lsa = tf.gather_nd(y_pred, prediction_indices)

        y_true_boxes_lsa = y_true_lsa[..., :-1]
        y_true_labels_lsa = y_true_lsa[..., -1]
        y_pred_boxes_lsa = y_pred_lsa[..., :4]
        y_pred_logits = y_pred[..., 4:]  # [1]

        batch_size = tf.shape(y_true)[0]
        n_pred_boxes = tf.shape(y_pred)[1]
        n_class = tf.shape(y_pred)[2] - 4
        no_class_labels = tf.cast(tf.fill([batch_size, n_pred_boxes], n_class - 1), tf.float32)
        y_true_labels_lsa = tf.tensor_scatter_nd_update(no_class_labels, prediction_indices,
                                                        y_true_labels_lsa)  # [batch_size, n_pred_boxes]

        loss_ce = self.weighted_cross_entropy_loss(y_true_labels_lsa, y_pred_logits) * self.cross_ent_weight
        loss_l1 = self.l1_loss(y_true_boxes_lsa, y_pred_boxes_lsa) * self.l1_weight
        loss_giou = self.giou_loss(y_true_boxes_lsa, y_pred_boxes_lsa) * self.giou_weight

        return loss_ce, loss_l1, loss_giou

    def _compute_cost_matrix(self, y_true, y_pred, batch_mask):
        cost_matrix = tf.math.add_n([loss_fn(y_true, y_pred) * loss_weight
                                     for loss_fn, loss_weight in zip(self.lsa_losses, self.lsa_loss_weights)])

        cost_matrix_mask = self._compute_cost_matrix_mask(cost_matrix, batch_mask)
        cost_matrix_ragged = tf.ragged.boolean_mask(cost_matrix, cost_matrix_mask)

        return cost_matrix_ragged

    @staticmethod
    def _compute_cost_matrix_mask(cost_matrix, batch_mask):
        n_pred_boxes = tf.shape(cost_matrix)[1]
        cost_matrix_mask = tf.tile(tf.expand_dims(batch_mask, 1), [1, n_pred_boxes, 1])
        return cost_matrix_mask

    @staticmethod
    def _lsa_to_batch_indices(lsa_indices, batch_mask):
        sizes = tf.reduce_sum(tf.cast(batch_mask, tf.int32), axis=1)
        row_idx = repeat_indices(sizes)
        row_idx = tf.tile(tf.expand_dims(row_idx, -1), [1, 2])
        indcs = tf.stack([row_idx, lsa_indices], axis=0)

        prediction_idx = tf.transpose(indcs[:, :, 0])
        target_idx = tf.transpose(indcs[:, :, 1])
        return prediction_idx, target_idx

    def _get_batch_mask(self, y_true):
        """
        :param y_true:
        :return: [batch_size, y_true.shape[0]]
        :rtype:
        """
        return tf.reduce_all(tf.not_equal(y_true, self.mask_value), -1)

    def get_config(self):
        config = {"n_classes": self.n_classes, "loss_weights": self.loss_weights,
                  "lsa_loss_weights": self.lsa_loss_weights, "no_class_weight": self.no_class_weight,
                  "mask_value": self.mask_value, "sequence_input": self.sequence_input, "sum_losses": self.sum_losses}
        base_config = super(HungarianLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedSparseCategoricalCrossEntropyCocoDETR(tf.keras.losses.Loss):
    def __init__(self, n_classes, no_class_weight=0.1, reduction=tf.keras.losses.Reduction.AUTO,
                 name="weighted_sparse_categorical_cross_entropy"):
        super().__init__(reduction=reduction, name=name)
        self.no_class_weight = no_class_weight
        self.n_classes = n_classes
        class_weights = tf.concat([tf.ones(self.n_classes, dtype=tf.float32),
                                   tf.constant([self.no_class_weight], dtype=tf.float32)],
                                  axis=0)
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.n_classes + 1)  # +1 for "no object" class

        weights = self.class_weights * y_true
        weights = tf.reduce_sum(weights, axis=-1)

        loss_ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss_ce = loss_ce * weights
        loss_ce = tf.reduce_sum(loss_ce) / tf.reduce_sum(weights)
        return loss_ce

    def get_config(self):
        config = {"n_classes": self.n_classes, "no_class_weight": self.no_class_weight}
        base_config = super(WeightedSparseCategoricalCrossEntropyCocoDETR, self).get_config()
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
