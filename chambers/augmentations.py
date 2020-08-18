import tensorflow as tf

from chambers.utils.tf import resize as _resize


def random_size_crop(img, boxes, min_size, max_size):
    hw = tf.random.uniform([2], min_size, max_size + 1, dtype=tf.int32)
    h = hw[0]
    w = hw[1]

    img = tf.image.random_crop(img, (h, w, 3))

    # TODO: Crop boxes

    return img, boxes


@tf.function
def flip_up_down(img, boxes):
    """
    Flip an image and bounding boxes vertically (upside down).
    :param img: 3-D Tensor of shape [height, width, channels]
    :param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [y0, x0, y0, x0]
    :return: image, bounding boxes
    """
    h = tf.cast(tf.shape(img)[0], tf.float32)
    with tf.name_scope("flip_up_down"):
        # bboxes = bboxes * tf.constant([-1, 1, -1, 1], dtype=tf.float32) + tf.stack([1.0, 0.0, 1.0, 0.0])
        boxes = boxes * tf.constant([-1, 1, -1, 1], dtype=tf.float32) + tf.stack([h, 0.0, h, 0.0])
        boxes = tf.stack([boxes[:, 2], boxes[:, 1], boxes[:, 0], boxes[:, 3]], axis=1)

        img = tf.image.flip_up_down(img)

    return img, boxes


@tf.function
def flip_left_right(img, boxes):
    """
    Flip an image and bounding boxes horizontally (left to right).
    :param img: 3-D Tensor of shape [height, width, channels]
    :param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [y0, x0, y0, x0]
    :return: image, bounding boxes
    """
    w = tf.cast(tf.shape(img)[1], tf.float32)
    with tf.name_scope("flip_left_right"):
        # bboxes = bboxes * tf.constant([1, -1, 1, -1], dtype=tf.float32) + tf.stack([0.0, 1.0, 0.0, 1.0])
        boxes = boxes * tf.constant([1, -1, 1, -1], dtype=tf.float32) + tf.stack([0.0, w, 0.0, w])
        boxes = tf.stack([boxes[:, 0], boxes[:, 3], boxes[:, 2], boxes[:, 1]], axis=1)

        img = tf.image.flip_left_right(img)

    return img, boxes


def resize(img, boxes, min_side=800, max_side=1333):
    """


    :param img: image with shape [h, w, c]
    :type img:
    :param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [y0, x0, y1, x1]
    :type boxes:
    :param min_side:
    :type min_side:
    :param max_side:
    :type max_side:
    :return:
    :rtype:
    """
    imgr = _resize(img, min_side=min_side, max_side=max_side)

    img_hw = tf.shape(img)[:2]
    imgr_hw = tf.shape(imgr)[:2]

    hw_ratios = tf.cast(imgr_hw / img_hw, tf.float32)
    h_ratio = hw_ratios[0]
    w_ratio = hw_ratios[1]

    boxr = boxes * tf.stack([h_ratio, w_ratio, h_ratio, w_ratio])  # [y0, x0, y1, x1]

    # [x0, y0, x1, y1], or [x0, y0, w, h] or [center_x, center_y, w, h]
    # boxr = boxes * tf.stack([w_ratio, h_ratio, w_ratio, h_ratio])

    return imgr, boxr


def random_resize_min(img, boxes, min_sides, max_side=1333):
    min_sides = tf.convert_to_tensor(min_sides)

    rand_idx = tf.random.uniform([1], minval=0, maxval=tf.shape(min_sides)[0], dtype=tf.int32)[0]
    min_side = min_sides[rand_idx]
    return resize(img, boxes, min_side=min_side, max_side=max_side)


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
    return _box_normalize_xyxy(boxes, h, w)


def box_normalize_cxcywh(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_normalize_xyxy(boxes, h, w)


def box_denormalize_xyxy(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_denormalize_xyxy(boxes, h, w)


def box_denormalize_yxyx(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_denormalize_yxyx(boxes, h, w)


def box_denormalize_xywh(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_denormalize_xyxy(boxes, h, w)


def box_denormalize_cxcywh(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_denormalize_xyxy(boxes, h, w)


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


def _box_denormalize_xyxy(boxes, img_h, img_w):
    """
    Denormalizes bounding box with coordinates between 0 and 1 to have coordinates in original image height and width.
    Bounding boxes are expected to have format [x0, y0, x1, y1].

    :param boxes: List of bounding boxes each with format [x0, y0, x1, y1]
    :type boxes: list[list] or tensorflow.Tensor
    :param img_h: The height of the image the bounding box belongs to.
    :type img_h: int or tensorflow.Tensor
    :param img_w: The width of the image the bounding box belongs to.
    :type img_w: int or tensorflow.Tensor
    :return: Normalized bounding boxes.
    :rtype: tensorflow.Tensor
    """

    boxes = boxes * tf.cast(tf.stack([img_w, img_h, img_w, img_h]), tf.float32)  # boxes * [w, h, w, h]
    return boxes


def _box_denormalize_yxyx(boxes, img_h, img_w):
    """
    Denormalizes bounding box with coordinates between 0 and 1 to have coordinates in original image height and width.
    Bounding boxes are expected to have format [y0, x0, y1, x1].

    :param boxes: List of bounding boxes each with format [y0, x0, y1, x1]
    :type boxes: list[list] or tensorflow.Tensor
    :param img_h: The height of the image the bounding box belongs to.
    :type img_h: int or tensorflow.Tensor
    :param img_w: The width of the image the bounding box belongs to.
    :type img_w: int or tensorflow.Tensor
    :return: Normalized bounding boxes.
    :rtype: tensorflow.Tensor
    """

    boxes = boxes * tf.cast(tf.stack([img_h, img_w, img_h, img_w]), tf.float32)  # boxes * [h, w, h, w]
    return boxes
