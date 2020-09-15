import tensorflow as tf

from chambers.utils.boxes import boxes_giou, box_yxyx_to_xyxy
from chambers.utils.tf import pairwise_l1 as _pairwise_l1
from chambers.utils.utils import deserialize_object


def pairwise_giou(y_true, y_pred):
    """
    :param y_true: (batch_size, n_true_boxes, 5)
    :param y_pred: (batch_size, n_pred_boxes, 4 + n_classes)
    :return: (batch_size, n_pred_boxes, n_true_boxes)
    """

    y_true = y_true[..., :-1]  # [batch_size, n_true_boxes, 4]
    y_pred = y_pred[..., :4]  # [batch_size, n_pred_boxes, 4]

    # giou cost
    y_true = box_yxyx_to_xyxy(y_true)
    y_pred = box_yxyx_to_xyxy(y_pred)
    cost_giou = tf.vectorized_map(lambda inp: -boxes_giou(*inp), (y_true, y_pred))
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


def pairwise_softmax(y_true, y_pred):
    """
    :param y_true: (batch_size, n_true_boxes, 5)
    :param y_pred: (batch_size, n_pred_boxes, 4 + n_classes)
    :return: (batch_size, n_pred_boxes, n_true_boxes)
    """

    y_true = y_true[..., -1]
    y_pred = y_pred[..., 4:]
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    # tf.gather does not take indices that are out of bounds when using CPU. So converting out of bounds indices to 0.
    n_class = tf.cast(tf.shape(y_pred)[-1], y_true.dtype)
    out_of_bound = tf.logical_or(tf.less(y_true, 0), tf.greater(y_true, n_class))
    y_true = tf.where(out_of_bound, tf.zeros_like(y_true), y_true)
    y_true = tf.cast(y_true, tf.int32)

    return tf.vectorized_map(lambda inp: -tf.gather(*inp, axis=-1), (y_pred, y_true))


def get(identifier, **kwargs):
    if type(identifier) is str:
        module_objs = globals()
        return deserialize_object(identifier,
                                  module_objects=module_objs,
                                  module_name=module_objs.get("__name__"),
                                  **kwargs
                                  )
    elif issubclass(identifier.__class__, type):
        return identifier(**kwargs)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(
            'Could not interpret encoder model identifier: {}'.format(identifier))
