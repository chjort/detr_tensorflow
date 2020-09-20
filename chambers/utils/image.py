import tensorflow as tf


def read_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image, channels=3)
    return image


def read_jpeg(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def resize(image, min_side=800, max_side=1333):
    """
    image with rank 3 [h, w, c]
    """

    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)

    cur_min_side = tf.minimum(w, h)
    min_side = tf.cast(min_side, tf.float32)

    if max_side is not None:
        cur_max_side = tf.maximum(w, h)
        max_side = tf.cast(max_side, tf.float32)
        scale = tf.minimum(max_side / cur_max_side,
                           min_side / cur_min_side)
    else:
        scale = min_side / cur_min_side

    nh = tf.cast(scale * h, tf.int32)
    nw = tf.cast(scale * w, tf.int32)

    image = tf.image.resize(image, (nh, nw))
    return image


def denormalize_image(image):
    image = tf.cast(image, dtype=tf.float32)
    channel_avg = tf.constant([0.485, 0.456, 0.406])
    channel_std = tf.constant([0.229, 0.224, 0.225])

    image = image * channel_std
    image = image + channel_avg
    image = image * 255.0
    image = tf.cast(image, tf.uint8)
    return image


def normalize_image(image):
    image = tf.cast(image, dtype=tf.float32)
    channel_avg = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    channel_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    image = (image / 255.0 - channel_avg) / channel_std
    return image


def resnet_imagenet_normalize(x):
    x = tf.cast(x, tf.float32)

    # RGB -> BGR
    x = x[..., ::-1]
    x = normalize_image(x, mean=[103.939, 116.779, 123.68])

    return x