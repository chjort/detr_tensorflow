import tensorflow as tf

from chambers.utils.boxes import boxes_giou, box_yxyx_to_xyxy
from chambers.utils.utils import deserialize_object


def pairwise_l2(a, b):
    """ L2 Distance (Euclidean distance) """
    x = pairwise_subtract(a, b)
    x = tf.square(x)
    x = tf.reduce_sum(x, -1)
    x = tf.sqrt(x)
    return x


def pairwise_l1(a, b):
    """ L1 Distance (Manhattan distance) """
    x = pairwise_subtract(a, b)
    x = tf.abs(x)
    x = tf.reduce_sum(x, -1)
    return x


def pairwise_giou(y_true, y_pred):
    """
    :param y_true: (batch_size, n_true_boxes, 4)
    :param y_pred: (batch_size, n_pred_boxes, 4)
    :return: (batch_size, n_pred_boxes, n_true_boxes)
    """

    y_true = box_yxyx_to_xyxy(y_true)
    y_pred = box_yxyx_to_xyxy(y_pred)
    cost_giou = tf.vectorized_map(lambda inp: -boxes_giou(*inp), (y_true, y_pred))
    return cost_giou


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


def pairwise_subtract(a, b):
    tf.debugging.assert_rank(b, tf.rank(a), "Tensor 'b' must have same rank as tensor 'a'")

    a_dim = tf.rank(a) - 1
    b_dim = tf.rank(b) - 2
    a = tf.expand_dims(a, axis=a_dim)
    b = tf.expand_dims(b, axis=b_dim)
    return a - b