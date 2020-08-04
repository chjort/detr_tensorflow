import tensorflow as tf


def apply_mask_image(image, mask):
    """ Expects image shape [h, w, c] and mask shape [h, w] """
    x = tf.ragged.boolean_mask(image, tf.logical_not(mask))

    is_not_padding_row = tf.logical_not(tf.equal(x.nested_row_lengths(), 0))
    height = tf.reduce_sum(tf.cast(is_not_padding_row, tf.int32))
    width = x.bounding_shape()[1]
    channels = x.bounding_shape()[2]

    x = x.to_tensor(shape=(height, width, channels))
    return x


def apply_mask_box(box, mask):
    """ Expects box shape [n_boxes, 4] and mask shape [n_boxes] """
    x = tf.ragged.boolean_mask(box, tf.logical_not(mask))
    return x


def remove_padding_image(image, padding_value):
    """ Expects image shape [h, w, c] """
    mask = tf.reduce_all(tf.equal(image, padding_value), axis=-1)
    return apply_mask_image(image, mask)


def remove_padding_box(box, padding_value):
    """ Expects box shape [n_boxes, 4] """
    mask = tf.reduce_all(tf.equal(box, padding_value), axis=1)
    return apply_mask_box(box, mask)
