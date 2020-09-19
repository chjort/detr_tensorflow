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
