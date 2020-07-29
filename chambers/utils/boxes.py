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


def boxes_resize_guard(boxes, img_h, img_w):
    """
    TODO

    :param boxes: Bounding boxes with format [x, y, w, h]
    :type boxes: tensorflow.Tensor
    :param img_h: Height of image the bounding boxes belong to.
    :type img_h: int
    :param img_w: Widht of image the bounding boxes belong to.
    :type img_w: int
    :return: Bounding boxes TODO
    :rtype: tensorflow.Tensor
    """

    nboxes = tf.shape(boxes)[0]
    row_idx = tf.range(nboxes)
    x_coords = tf.stack([row_idx, tf.repeat(0, nboxes)], axis=1)
    y_coords = tf.stack([row_idx, tf.repeat(1, nboxes)], axis=1)
    w_coords = tf.stack([row_idx, tf.repeat(2, nboxes)], axis=1)
    h_coords = tf.stack([row_idx, tf.repeat(3, nboxes)], axis=1)

    xw_coords = tf.concat([x_coords, w_coords], axis=0)
    yh_coords = tf.concat([y_coords, h_coords], axis=0)
    xy_coords = tf.concat([x_coords, y_coords], axis=0)
    wh_coords = tf.concat([w_coords, h_coords], axis=0)

    xw_values = tf.gather_nd(boxes, xw_coords)  # boxes[:, 0::2]
    yh_values = tf.gather_nd(boxes, yh_coords)  # boxes[:, 1::2]
    xy_values = tf.gather_nd(boxes, xy_coords)  # boxes[:, :2]

    boxes = tf.tensor_scatter_nd_add(boxes, wh_coords, xy_values) # boxes[:, 2:] += boxes[:, :2]

    # zero = tf.constant(0, dtype=tf.float32)
    # img_w = tf.cast(img_w, tf.float32)
    # img_h = tf.cast(img_h, tf.float32)
    # boxes = tf.tensor_scatter_nd_update(boxes, xw_coords, tf.clip_by_value(xw_values, zero, img_w))
    # boxes = tf.tensor_scatter_nd_update(boxes, yh_coords, tf.clip_by_value(yh_values, zero, img_h))

    return boxes


def normalize_boxes(boxes, img_h, img_w):
    nboxes = tf.shape(boxes)[0]
    row_idx = tf.range(nboxes)
    x_coords = tf.stack([row_idx, tf.repeat(0, nboxes)], axis=1)
    y_coords = tf.stack([row_idx, tf.repeat(1, nboxes)], axis=1)
    w_coords = tf.stack([row_idx, tf.repeat(2, nboxes)], axis=1)
    h_coords = tf.stack([row_idx, tf.repeat(3, nboxes)], axis=1)

    xw_coords = tf.concat([x_coords, w_coords], axis=0)
    yh_coords = tf.concat([y_coords, h_coords], axis=0)
    xy_coords = tf.concat([x_coords, y_coords], axis=0)
    # wh_coords = tf.concat([w_coords, h_coords], axis=0)

    xw_values = tf.gather_nd(boxes, xw_coords)
    yh_values = tf.gather_nd(boxes, yh_coords)
    # xy_values = tf.gather_nd(boxes, xy_coords)

    img_w = tf.cast(img_w, tf.float32)
    img_h = tf.cast(img_h, tf.float32)
    boxes = tf.tensor_scatter_nd_update(boxes, xw_coords, xw_values / img_w)
    boxes = tf.tensor_scatter_nd_update(boxes, yh_coords, yh_values / img_h)
    return boxes