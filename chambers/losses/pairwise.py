import tensorflow as tf

from chambers.utils.boxes import box_yxyx_to_xyxy, box_area
from chambers.utils.utils import deserialize_object


def pairwise_subtract(y_true, y_pred):
    tf.debugging.assert_rank(y_pred, tf.rank(y_true), "Tensor 'y_pred' must have same rank as tensor 'y_true'")

    y_true_dim = tf.rank(y_true) - 1
    y_pred_dim = tf.rank(y_pred) - 2
    y_true = tf.expand_dims(y_true, axis=y_true_dim)
    y_pred = tf.expand_dims(y_pred, axis=y_pred_dim)
    return y_true - y_pred


def pairwise_l2(y_true, y_pred):
    """ L2 Distance (Euclidean distance) """
    l2_diff = pairwise_subtract(y_true, y_pred)
    l2_diff = tf.square(l2_diff)
    l2_diff = tf.reduce_sum(l2_diff, -1)
    l2_diff = tf.sqrt(l2_diff)
    return l2_diff


def pairwise_l1(y_true, y_pred):
    """ L1 Distance (Manhattan distance) """
    l1_diff = pairwise_subtract(y_true, y_pred)
    l1_diff = tf.abs(l1_diff)
    l1_diff = tf.reduce_sum(l1_diff, -1)
    return l1_diff


def pairwise_giou(y_true, y_pred):
    """
    :param y_true: (batch_size, n_true_boxes, 4)
    :param y_pred: (batch_size, n_pred_boxes, 4)
    :return: (batch_size, n_pred_boxes, n_true_boxes)
    """

    y_true = box_yxyx_to_xyxy(y_true)
    y_pred = box_yxyx_to_xyxy(y_pred)
    iou = tf.vectorized_map(lambda inp: -_boxes_iou(*inp), (y_true, y_pred))
    return iou


def pairwise_giou(y_true, y_pred):
    """
    :param y_true: (batch_size, n_true_boxes, 4)
    :param y_pred: (batch_size, n_pred_boxes, 4)
    :return: (batch_size, n_pred_boxes, n_true_boxes)
    """

    y_true = box_yxyx_to_xyxy(y_true)
    y_pred = box_yxyx_to_xyxy(y_pred)
    giou = tf.vectorized_map(lambda inp: -_boxes_giou(*inp), (y_true, y_pred))
    return giou


def _boxes_iou(y_true, y_pred):
    """
    Computes IOU between predicted bounding boxes and ground truth bounding boxes.
    The bounding boxes are expected to have format [x0, y0, x1, y1].

    :param y_true: Ground truth bounding box with format [x0, y0, x1, y1].
    :type y_true: tensorflow.Tensor
    :param y_pred: Predicted bounding box with format [x0, y0, x1, y1].
    :type y_pred: tensorflow.Tensor
    :return: Intersection over union between y_pred and y_true
    :rtype: tensorflow.Tensor
    """
    b1_area = box_area(y_pred)
    b2_area = box_area(y_true)

    left_top = tf.maximum(y_pred[:, None, :2], y_true[:, :2])
    right_bottom = tf.minimum(y_pred[:, None, 2:], y_true[:, 2:])

    wh = right_bottom - left_top
    wh = tf.clip_by_value(wh, 0, tf.reduce_max(wh))

    intersection = wh[..., 0] * wh[..., 1]
    union = b1_area[:, None] + b2_area - intersection
    iou = intersection / union

    return iou, union


def _boxes_giou(y_true, y_pred):
    """
    Computes Generalized IOU between predicted bounding boxes and ground truth bounding boxes.
    The bounding boxes are expected to have format [x0, y0, x1, y1].

    :param y_true: Ground truth bounding box with format [x0, y0, x1, y1].
    :type y_true: tensorflow.Tensor
    :param y_pred: Predicted bounding box with format [x0, y0, x1, y1].
    :type y_pred: tensorflow.Tensor
    :return: Generalized intersection over union between y_pred and y_true
    :rtype: tensorflow.Tensor
    """
    iou, union = _boxes_iou(y_true, y_pred)

    left_bottom = tf.minimum(y_pred[:, None, :2], y_true[:, :2])
    right_top = tf.maximum(y_pred[:, None, 2:], y_true[:, 2:])

    wh = right_top - left_bottom
    wh = tf.clip_by_value(wh, 0, tf.reduce_max(wh))

    area = wh[..., 0] * wh[..., 1]

    giou = iou - (area - union) / area
    return giou


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
