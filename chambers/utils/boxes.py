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
def boxes_iou(boxes1, boxes2):
    b1_area = box_area(boxes1)
    b2_area = box_area(boxes2)

    left_top = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = right_bottom - left_top
    wh = tf.clip_by_value(wh, 0, tf.reduce_max(wh))

    intersection = wh[..., 0] * wh[..., 1]
    union = b1_area[:, None] + b2_area - intersection
    iou = intersection / union

    return iou, union


def boxes_giou(boxes1, boxes2):
    iou, union = boxes_iou(boxes1, boxes2)

    left_bottom = tf.minimum(boxes1[:, None, :2], boxes2[:, :2])
    right_top = tf.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = right_top - left_bottom
    wh = tf.clip_by_value(wh, 0, tf.reduce_max(wh))

    area = wh[..., 0] * wh[..., 1]

    giou = iou - (area - union) / area
    return giou


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = tf.unstack(x, 4, axis=1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return tf.stack(b, axis=-1)


def box_cxcywh_to_yxyx(x):
    x_c, y_c, w, h = tf.unstack(x, 4, axis=1)
    b = [(y_c - 0.5 * h), (x_c - 0.5 * w),
         (y_c + 0.5 * h), (x_c + 0.5 * w)]
    return tf.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = tf.unstack(x, 4, axis=1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return tf.stack(b, axis=-1)
