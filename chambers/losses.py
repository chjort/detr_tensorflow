import tensorflow as tf

from chambers.utils.boxes import box_cxcywh_to_xyxy, boxes_giou
from chambers.utils.tf import pairwise_l1, batch_linear_sum_assignment, repeat_indices


class HungarianLoss(tf.keras.losses.Loss):
    """
    Computes the Linear Sum Assignment (LSA) between predictions and targets, using a weighted sum of L1,
    Generalized IOU and softmax probabilities for the cost matrix.

    The final loss is given as a weighted sum of cross-entropy, L1 and Generalized IOU on the prediction-target pairs
    given by the LSA.

    """

    def __init__(self, mask_value=None, sequence_input=False, name="hungarian_loss"):
        self.mask_value = mask_value
        self.sequence_input = sequence_input
        self.class_loss_weight = 1
        self.bbox_loss_weight = 5
        self.giou_loss_weight = 2

        if sequence_input:
            self.input_signature = [tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32),
                                    tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)]
        else:
            self.input_signature = [tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32),
                                    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)]
        self.call = tf.function(self.call, input_signature=self.input_signature)

        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)

    def call(self, y_true, y_pred):
        y_true_boxes = y_true[..., :-1]  # [0]
        y_true_labels = y_true[..., -1]  # [1]
        y_pred_boxes = y_pred[..., :4]   # [0]
        y_pred_logits = y_pred[..., 4:]  # [1]

        if self.sequence_input:
            tf.assert_rank(y_pred_boxes, 4, "Invalid input shape.")
            tf.assert_rank(y_pred_logits, 4, "Invalid input shape.")

            seq_len = tf.shape(y_pred_logits)[1]
            loss = tf.constant(0, tf.float32)
            for i in tf.range(seq_len):
                y_pred_logits_i = y_pred_logits[:, i, :, :]
                y_pred_boxes_i = y_pred_boxes[:, i, :, :]
                loss_i = self._compute_loss(y_true_labels, y_true_boxes, y_pred_logits_i, y_pred_boxes_i)
                loss = loss + loss_i
        else:
            loss = self._compute_loss(y_true_labels, y_true_boxes, y_pred_logits, y_pred_boxes)

        return loss

    def _compute_loss(self, y_true_labels, y_true_boxes, y_pred_logits, y_pred_boxes):
        """


        :param y_true_labels: [batch_size, n_true_boxes]
        :type y_true_labels:
        :param y_true_boxes: [batch_size, n_true_boxes, 4]
        :type y_true_boxes:
        :param y_pred_logits: [batch_size, n_pred_boxes, n_classes]
        :type y_pred_logits:
        :param y_pred_boxes: [batch_size, n_pred_boxes, 4]
        :type y_pred_boxes:
        :return:
        :rtype:
        """
        batch_size = tf.shape(y_pred_logits)[0]
        n_pred_boxes = tf.shape(y_pred_logits)[1]
        n_class = tf.shape(y_pred_logits)[2]
        batch_mask = self._get_batch_mask(y_true_boxes)  # [batch_size, max_n_true_boxes_batch]

        # cost_matrix (RaggedTensor) [batch_size, n_pred_boxes, None]
        cost_matrix = self._compute_cost_matrix(y_true_labels, y_true_boxes, y_pred_logits, y_pred_boxes, batch_mask)

        lsa = batch_linear_sum_assignment(cost_matrix)  # [n_true_boxes, 2]

        prediction_indices, target_indices = self._lsa_to_gather_indices(lsa,
                                                                         batch_mask)  # ([n_true_boxes, 2], [n_true_boxes, 2])

        # get assigned targets
        y_true_boxes_lsa = tf.gather_nd(y_true_boxes, target_indices)  # [n_true_boxes, 4]
        y_true_labels_lsa = tf.gather_nd(y_true_labels, target_indices)
        no_class_labels = tf.cast(tf.fill([batch_size, n_pred_boxes], n_class - 1), tf.float32)
        y_true_labels_lsa = tf.tensor_scatter_nd_update(no_class_labels, prediction_indices,
                                                        y_true_labels_lsa)  # [batch_size, n_pred_boxes]

        # get assigned predictions
        y_pred_boxes_lsa = tf.gather_nd(y_pred_boxes, prediction_indices)  # [n_true_boxes, 4]

        loss_ce = self.weighted_cross_entropy_loss(y_true_labels_lsa, y_pred_logits) * self.class_loss_weight
        loss_l1 = self.l1_loss(y_true_boxes_lsa, y_pred_boxes_lsa) * self.bbox_loss_weight
        loss_giou = self.giou_loss(y_true_boxes_lsa, y_pred_boxes_lsa) * self.giou_loss_weight

        return loss_ce + loss_l1 + loss_giou

    @staticmethod
    def weighted_cross_entropy_loss(y_true, y_pred):
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

    @staticmethod
    def l1_loss(y_true, y_pred):
        n_target_boxes = tf.cast(tf.shape(y_true)[0], tf.float32)
        loss_l1_box = tf.reduce_sum(tf.abs(y_true - y_pred)) / n_target_boxes
        return loss_l1_box

    @staticmethod
    def giou_loss(y_true, y_pred):
        y_true = box_cxcywh_to_xyxy(y_true)
        y_pred = box_cxcywh_to_xyxy(y_pred)
        giou = boxes_giou(y_true, y_pred)
        giou = tf.linalg.diag_part(giou)
        loss_giou_box = tf.reduce_mean(1 - giou)
        return loss_giou_box

    def _compute_label_cost(self, y_true, y_pred):
        """
        (batch_size, n_true_boxes), (batch_size, n_pred_boxes, n_classes)
        """

        n_class = tf.shape(y_pred)[2]

        # labels
        y_pred = tf.reshape(y_pred, [-1, n_class])
        # y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = tf.nn.log_softmax(y_pred, axis=-1)

        y_true = tf.reshape(y_true, [-1])

        # logits cost
        # TODO: tf.gather does not take indices that are out of bounds when using CPU? Works on Azure VM with GPU.
        y_true = tf.where(tf.equal(y_true, self.mask_value), tf.zeros_like(y_true), y_true)
        cost_class = -tf.gather(y_pred, tf.cast(y_true, tf.int32), axis=1)
        return cost_class

    def _compute_box_l1_cost(self, y_true, y_pred):
        """
        (batch_size, n_true_boxes, 4), (batch_size, n_pred_boxes, 4)
        """

        box_shape = tf.shape(y_pred)[-1]
        y_pred = tf.reshape(y_pred, [-1, box_shape])

        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])

        # bbox cost
        cost_bbox = pairwise_l1(y_pred, y_true)
        return cost_bbox

    def _compute_box_giou_cost(self, y_true, y_pred):
        """
        (batch_size, n_true_boxes, 4), (batch_size, n_pred_boxes, 4)
        """

        box_shape = tf.shape(y_pred)[-1]
        y_pred = tf.reshape(y_pred, [-1, box_shape])

        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])

        # giou cost
        y_true = box_cxcywh_to_xyxy(y_true)
        y_pred = box_cxcywh_to_xyxy(y_pred)
        cost_giou = -boxes_giou(y_true, y_pred)
        return cost_giou

    def _compute_cost_matrix(self, y_true_logits, y_true_boxes, y_pred_logits, y_pred_boxes, batch_mask):
        batch_size = tf.shape(y_pred_logits)[0]
        n_pred_boxes = tf.shape(y_pred_logits)[1]

        cost_class = self._compute_label_cost(y_true_logits, y_pred_logits)
        cost_bbox = self._compute_box_l1_cost(y_true_boxes, y_pred_boxes)
        cost_giou = self._compute_box_giou_cost(y_true_boxes, y_pred_boxes)

        cost_matrix = self.bbox_loss_weight * cost_bbox + self.class_loss_weight * cost_class + self.giou_loss_weight * cost_giou
        cost_matrix = tf.reshape(cost_matrix, [batch_size, n_pred_boxes, -1])

        cost_matrix_mask = self._compute_cost_matrix_mask(cost_matrix, batch_mask)

        cost_matrix_ragged = tf.ragged.boolean_mask(cost_matrix, cost_matrix_mask)

        return cost_matrix_ragged

    def _compute_cost_matrix_mask(self, cost_matrix, batch_mask):
        batch_size = tf.shape(cost_matrix)[0]
        n_box = tf.shape(cost_matrix)[1]
        batch_mask_flat = tf.reshape(batch_mask, [-1])

        mask_mask = tf.repeat(tf.eye(batch_size, batch_size), tf.shape(batch_mask)[1])
        C_mask = tf.cast(tf.tile(batch_mask_flat, [batch_size]), tf.float32) * tf.cast(mask_mask, tf.float32)
        C_mask = tf.cast(tf.reshape(C_mask, [batch_size, -1]), tf.bool)
        C_mask = tf.tile(tf.expand_dims(C_mask, 1), [1, n_box, 1])
        return C_mask

    def _get_batch_mask(self, x_padded):
        """

        :param x_padded:
        :type x_padded:
        :return: [batch_size, x_padded.shape[0]]
        :rtype:
        """
        return tf.reduce_all(tf.not_equal(x_padded, self.mask_value), -1)

    def _lsa_to_gather_indices(self, lsa_indices, batch_mask):
        sizes = tf.reduce_sum(tf.cast(batch_mask, tf.int32), axis=1)
        row_idx = repeat_indices(sizes)
        row_idx = tf.tile(tf.expand_dims(row_idx, -1), [1, 2])
        indcs = tf.stack([row_idx, lsa_indices], axis=0)

        prediction_idx = tf.transpose(indcs[:, :, 0])
        target_idx = tf.transpose(indcs[:, :, 1])
        return prediction_idx, target_idx