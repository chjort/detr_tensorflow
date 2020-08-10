import tensorflow as tf

from chambers.utils.tf import resize as _resize


def resize(img, box, min_side=800, max_side=1333):
    imgr = _resize(img, min_side=min_side, max_side=max_side)

    img_hw = tf.shape(img)[:2]
    imgr_hw = tf.shape(imgr)[:2]

    hw_ratios = tf.cast(imgr_hw / img_hw, tf.float32)
    h_ratio = hw_ratios[0]
    w_ratio = hw_ratios[1]
    # boxr = box * tf.stack([h_ratio, w_ratio, h_ratio, w_ratio])
    boxr = box * tf.stack([w_ratio, h_ratio, w_ratio, h_ratio])

    return imgr, boxr


def random_resize_min(img, box, min_sides, max_side=1333):
    min_sides = tf.convert_to_tensor(min_sides)

    rand_idx = tf.random.uniform([1], minval=0, maxval=tf.shape(min_sides)[0], dtype=tf.int32)[0]
    min_side = min_sides[rand_idx]
    return resize(img, box, min_side=min_side, max_side=max_side)


def box_normalize_xyxy(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_normalize_xyxy(boxes, h, w)


def box_normalize_yxyx(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_normalize_yxyx(boxes, h, w)


def box_normalize_xywh(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_normalize_xywh(boxes, h, w)


def box_normalize_cxcywh(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_normalize_cxcywh(boxes, h, w)


def _box_normalize_xyxy(boxes, img_h, img_w):
    """
    Normalizes bounding box to have coordinates between 0 and 1. Bounding boxes are expected to
    have format [x0, y0, x1, y1].

    :param boxes: List of bounding boxes each with format [x0, y0, x1, y1]
    :type boxes: list[list] or tensorflow.Tensor
    :param img_h: The height of the image the bounding box belongs to.
    :type img_h: int or tensorflow.Tensor
    :param img_w: The width of the image the bounding box belongs to.
    :type img_w: int or tensorflow.Tensor
    :return: Normalized bounding boxes.
    :rtype: tensorflow.Tensor
    """

    boxes = boxes / tf.cast(tf.stack([img_w, img_h, img_w, img_h]), tf.float32)  # boxes / [w, h, w, h]
    return boxes


def _box_normalize_yxyx(boxes, img_h, img_w):
    """
    Normalizes bounding box to have coordinates between 0 and 1. Bounding boxes are expected to
    have format [y0, x0, y1, x1].

    :param boxes: List of bounding boxes each with format [y0, x0, y1, x1]
    :type boxes: list[list] or tensorflow.Tensor
    :param img_h: The height of the image the bounding box belongs to.
    :type img_h: int or tensorflow.Tensor
    :param img_w: The width of the image the bounding box belongs to.
    :type img_w: int or tensorflow.Tensor
    :return: Normalized bounding boxes.
    :rtype: tensorflow.Tensor
    """

    boxes = boxes / tf.cast(tf.stack([img_h, img_w, img_h, img_w]), tf.float32)  # boxes / [h, w, h, w]
    return boxes


def _box_normalize_xywh(boxes, img_h, img_w):
    return _box_normalize_xyxy(boxes, img_h, img_w)


def _box_normalize_cxcywh(boxes, img_h, img_w):
    return _box_normalize_xyxy(boxes, img_h, img_w)
