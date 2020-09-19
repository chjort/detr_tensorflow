import tensorflow as tf


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# TODO: put in metrics
def boxes_iou(y_true, y_pred):
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


def boxes_giou(y_true, y_pred):
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
    iou, union = boxes_iou(y_true, y_pred)

    left_bottom = tf.minimum(y_pred[:, None, :2], y_true[:, :2])
    right_top = tf.maximum(y_pred[:, None, 2:], y_true[:, 2:])

    wh = right_top - left_bottom
    wh = tf.clip_by_value(wh, 0, tf.reduce_max(wh))

    area = wh[..., 0] * wh[..., 1]

    giou = iou - (area - union) / area
    return giou


### TO XYXY ###
def box_yxyx_to_xyxy(x):
    """ Converts bounding box with format [y0, x0, y1, x1] to format [x0, y0, x1, y1] """
    y0, x0, y1, x1 = tf.unstack(x, 4, axis=-1)
    b = [
        x0,
        y0,
        x1,
        y1
    ]
    return tf.stack(b, axis=-1)


def box_xywh_to_xyxy(x):
    """ Converts bounding box with format [x0, y0, w, h] to format [x0, y0, x1, y1] """
    x0, y0, w, h = tf.unstack(x, 4, axis=-1)
    b = [
        x0,
        y0,
        x0 + w,
        y0 + h
    ]
    return tf.stack(b, axis=-1)


def box_cxcywh_to_xyxy(x):
    """ Converts bounding box with format [center_x, center_y, w, h] to format [x0, y0, x1, y1] """
    cx, cy, w, h = tf.unstack(x, 4, axis=-1)
    b = [
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h
    ]
    return tf.stack(b, axis=-1)


### TO YXYX ###
def box_xyxy_to_yxyx(x):
    """ Converts bounding box with format [x0, y0, x1, y1] to format [y0, x0, y1, x1] """
    x0, y0, x1, y1 = tf.unstack(x, 4, axis=-1)
    b = [
        y0,
        x0,
        y1,
        x1
    ]
    return tf.stack(b, axis=-1)


def box_xywh_to_yxyx(x):
    """ Converts bounding box with format [x0, y0, w, h] to format [y0, x0, y1, x1] """
    x0, y0, w, h = tf.unstack(x, 4, axis=-1)
    b = [
        y0,
        x0,
        y0 + h,
        x0 + w
    ]
    return tf.stack(b, axis=-1)


def box_cxcywh_to_yxyx(x):
    """ Converts bounding box with format [center_x, center_y, w, h] to format [y0, x0, y1, x1] """
    cx, cy, w, h = tf.unstack(x, 4, axis=-1)
    b = [
        cy - 0.5 * h,
        cx - 0.5 * w,
        cy + 0.5 * h,
        cx + 0.5 * w
    ]
    return tf.stack(b, axis=-1)


### TO XYWH ###
def box_xyxy_to_xywh(x):
    """ Converts bounding box with format [x0, y0, x1, y1] to format [x0, y0, w, h] """
    x0, y0, x1, y1 = tf.unstack(x, 4, axis=-1)
    b = [
        x0,
        y0,
        x1 - x0,
        y1 - y0
    ]
    return tf.stack(b, axis=-1)


def box_yxyx_to_xywh(x):
    """ Converts bounding box with format [y0, x0, y1, x1] to format [x0, y0, w, h] """
    y0, x0, y1, x1 = tf.unstack(x, 4, axis=-1)
    b = [
        x0,
        y0,
        x1 - x0,
        y1 - y0
    ]
    return tf.stack(b, axis=-1)


def box_cxcywh_to_xywh(x):
    """ Converts bounding box with format [center_x, center_y, w, h] to format [x0, y0, w, h] """
    cx, cy, w, h = tf.unstack(x, 4, axis=-1)
    b = [
        cx - 0.5 * w,
        cy - 0.5 * h,
        w,
        h
    ]
    return tf.stack(b, axis=-1)


### TO CXCYWH ###
def box_xyxy_to_cxcywh(x):
    """ Converts bounding box with format [x0, y0, x1, y1] to format [center_x, center_y, w, h] """
    x0, y0, x1, y1 = tf.unstack(x, 4, axis=-1)
    b = [
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        x1 - x0,
        y1 - y0
    ]
    return tf.stack(b, axis=-1)


def box_yxyx_to_cxcywh(x):
    """ Converts bounding box with format [y0, x0, y1, x1] to format [center_x, center_y, w, h] """
    y0, x0, y1, x1 = tf.unstack(x, 4, axis=-1)
    b = [
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        x1 - x0,
        y1 - y0
    ]
    return tf.stack(b, axis=-1)


def box_xywh_to_cxcywh(x):
    """ Converts bounding box with format [x0, y0, w, h] to format [center_x, center_y, w, h] """
    x0, y0, w, h = tf.unstack(x, 4, axis=-1)
    b = [
        x0 + 0.5 * w,
        y0 + 0.5 * h,
        w,
        h
    ]
    return tf.stack(b, axis=-1)


def get(identifier):
    funcs = globals()
    if type(identifier) == str:
        f = funcs[identifier]
    else:
        raise ValueError("Argument 'identifier' must be type string.")

    return f


def relative_to_absolute(boxes, height, width):
    """
    :param boxes: Bounding boxes with format [y0, x0, y1, x1] and normalized between 0 and 1.
    :param height:
    :param width:
    :return:
    """
    scale = tf.constant([height, width, height, width], dtype=tf.float32)
    return boxes * scale
