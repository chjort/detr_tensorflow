import numpy as np
import tensorflow as tf

from chambers.utils.boxes import box_cxcywh_to_yxyx
from chambers.utils.tf import batch_linear_sum_assignment, repeat_indices
from .iou import GIoULoss
from .losses import L1Loss


class HungarianLoss(tf.keras.losses.Loss):
    """
    Computes the Linear Sum Assignment (LSA) between predictions and targets, using a weighted sum of L1,
    Generalized IOU and softmax probabilities for the cost matrix.

    The final loss is given as a weighted sum of cross-entropy, L1 and Generalized IOU on the prediction-target pairs
    given by the LSA.

    """

    def __init__(self, lsa_losses, lsa_loss_weights=None, mask_value=None, sequence_input=False, name="hungarian_loss"):
        self.mask_value = mask_value
        self.sequence_input = sequence_input
        self.lsa_losses = lsa_losses
        self.lsa_loss_weights = lsa_loss_weights

        self.weighted_cross_entropy_loss = WeightedSparseCategoricalCrossEntropyCocoDETR()
        self.l1_loss = L1Loss()
        self.giou_loss = GIoULoss()
        self.class_loss_weight = 1
        self.bbox_loss_weight = 5
        self.giou_loss_weight = 2

        # TODO: Gather losses from intermediate decoder layers when sequence_input=True
        # self.batch_loss_ce = tf.Variable(0, dtype=tf.float32, trainable=False)
        # self.batch_loss_l1 = tf.Variable(0, dtype=tf.float32, trainable=False)
        # self.batch_loss_giou = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.batch_losses = {
            "loss_ce": tf.Variable(0, dtype=tf.float32, trainable=False),
            "loss_l1": tf.Variable(0, dtype=tf.float32, trainable=False),
            "loss_giou": tf.Variable(0, dtype=tf.float32, trainable=False)
        }

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
        :type y_true:
        :param y_pred: shape [batch_size, n_pred_boxes, 4 + n_classes].
            If ``self.sequence_input`` is True shape must be [batch_size, sequence_len, n_true_boxes, 5]
        :type y_pred:
        :return:
        :rtype:
        """
        if self.sequence_input:
            tf.assert_rank(y_pred, 4, "Invalid input shape.")

            seq_len = tf.shape(y_pred)[1]
            loss = tf.constant(0, tf.float32)
            for i in tf.range(seq_len):
                y_pred_i = y_pred[:, i, :, :]
                # loss_i = self._compute_loss(y_true, y_pred_i)
                # loss = loss + loss_i
                losses_i = self._compute_losses(y_true, y_pred_i)
                loss = loss + tf.reduce_sum(losses_i)
        else:
            # loss = self._compute_loss(y_true, y_pred)
            losses = self._compute_losses(y_true, y_pred)
            loss = tf.reduce_sum(losses)

            for l, (k, v) in zip(losses, self.batch_losses.items()):
                v.assign(l)

        return loss

    def _compute_losses(self, y_true, y_pred):
        """

        :param y_true: [batch_size, n_true_boxes, 5]
        :type y_true:
        :param y_pred: [batch_size, n_pred_boxes, 4 + n_classes]
        :type y_pred:
        :return:
        :rtype:
        """

        batch_size = tf.shape(y_true)[0]
        n_pred_boxes = tf.shape(y_pred)[1]
        n_class = tf.shape(y_pred)[2] - 4
        batch_mask = self._get_batch_mask(y_true)  # [batch_size, max_n_true_boxes_batch]

        # cost_matrix (RaggedTensor) [batch_size, n_pred_boxes, None]
        cost_matrix = self._compute_cost_matrix(y_true, y_pred, batch_mask)  # TODO: input format

        # Ignore cost matrices that are all NaN.
        nan_matrices = tf.reduce_all(tf.math.is_nan(cost_matrix), axis=(1, 2))
        if tf.reduce_all(nan_matrices):
            return np.nan, np.nan, np.nan

        if tf.reduce_any(nan_matrices):
            no_nan_matrices = tf.logical_not(nan_matrices)
            cost_matrix = tf.ragged.boolean_mask(cost_matrix, no_nan_matrices)
            batch_mask = tf.cast(
                tf.transpose(tf.transpose(tf.cast(batch_mask, tf.float32)) * tf.cast(no_nan_matrices, tf.float32)),
                tf.bool)

        lsa = batch_linear_sum_assignment(cost_matrix)  # [n_true_boxes, 2]

        prediction_indices, target_indices = self._lsa_to_batch_indices(lsa, batch_mask)
        # ([n_true_boxes, 2], [n_true_boxes, 2])

        # get assigned targets
        y_true_boxes = y_true[..., :-1]  # [0]
        y_true_labels = y_true[..., -1]  # [1]
        y_pred_boxes = y_pred[..., :4]  # [0]
        y_pred_logits = y_pred[..., 4:]  # [1]

        y_true_boxes_lsa = tf.gather_nd(y_true_boxes, target_indices)  # [n_true_boxes, 4]
        y_true_labels_lsa = tf.gather_nd(y_true_labels, target_indices)
        no_class_labels = tf.cast(tf.fill([batch_size, n_pred_boxes], n_class - 1), tf.float32)
        y_true_labels_lsa = tf.tensor_scatter_nd_update(no_class_labels, prediction_indices,
                                                        y_true_labels_lsa)  # [batch_size, n_pred_boxes]

        # get assigned predictions
        y_pred_boxes_lsa = tf.gather_nd(y_pred_boxes, prediction_indices)  # [n_true_boxes, 4]

        # tf.print(tf.shape(y_true_labels_lsa), tf.shape(y_pred_logits), "-", tf.shape(y_true_boxes_lsa),
        #          tf.shape(y_pred_boxes_lsa))

        # TODO: Make these loss functions take ``y_true_lsa`` and ``y_pred_lsa`` as input
        loss_ce = self.weighted_cross_entropy_loss(y_true_labels_lsa, y_pred_logits) * self.class_loss_weight
        loss_l1 = self.l1_loss(y_true_boxes_lsa, y_pred_boxes_lsa) * self.bbox_loss_weight
        loss_giou = self.giou_loss(box_cxcywh_to_yxyx(y_true_boxes_lsa),
                                   box_cxcywh_to_yxyx(y_pred_boxes_lsa)) * self.giou_loss_weight

        # TODO: and then mask padded boxes/labels here

        # self.batch_loss_ce.assign(loss_ce)
        # self.batch_loss_l1.assign(loss_l1)
        # self.batch_loss_giou.assign(loss_giou)

        # loss = loss_ce + loss_l1 + loss_giou
        # return loss
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

    def _get_batch_mask(self, y_true):
        """
        :param y_true:
        :return: [batch_size, y_true.shape[0]]
        :rtype:
        """
        return tf.reduce_all(tf.not_equal(y_true, self.mask_value), -1)

    @staticmethod
    def _lsa_to_batch_indices(lsa_indices, batch_mask):
        sizes = tf.reduce_sum(tf.cast(batch_mask, tf.int32), axis=1)
        row_idx = repeat_indices(sizes)
        row_idx = tf.tile(tf.expand_dims(row_idx, -1), [1, 2])
        indcs = tf.stack([row_idx, lsa_indices], axis=0)

        prediction_idx = tf.transpose(indcs[:, :, 0])
        target_idx = tf.transpose(indcs[:, :, 1])
        return prediction_idx, target_idx


def weighted_cross_entropy_loss_coco_detr(y_true, y_pred):
    n_class = tf.shape(y_pred)[2]
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_class)

    real_class_weight = 1
    non_class_weight = 0.1
    class_weights = tf.concat([tf.repeat(tf.constant(real_class_weight, dtype=tf.float32), n_class - 1),
                               tf.constant([non_class_weight], dtype=tf.float32)],
                              axis=0)

    weights = class_weights * y_true
    weights = tf.reduce_sum(weights, axis=-1)

    loss_ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
    loss_ce = loss_ce * weights
    loss_ce = tf.reduce_sum(loss_ce) / tf.reduce_sum(weights)
    return loss_ce


class WeightedSparseCategoricalCrossEntropyCocoDETR(tf.keras.losses.Loss):
    def __init__(self, name="weighted_sparse_categorical_cross_entropy"):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        real_class_weight = 1
        non_class_weight = 0.1
        self.n_class = 92
        class_weights = tf.concat([tf.repeat(tf.constant(real_class_weight, dtype=tf.float32), self.n_class - 1),
                                   tf.constant([non_class_weight], dtype=tf.float32)],
                                  axis=0)
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.n_class)

        weights = self.class_weights * y_true
        weights = tf.reduce_sum(weights, axis=-1)

        loss_ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss_ce = loss_ce * weights
        loss_ce = tf.reduce_sum(loss_ce) / tf.reduce_sum(weights)
        return loss_ce
