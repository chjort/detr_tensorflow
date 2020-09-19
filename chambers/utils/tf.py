import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as linear_sum_assignment_scipy


def read_jpeg(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def set_supports_masking(model, verbose=True, **kwargs):
    """
        Sets the attribute 'supports_masking' to True for every layer in model.
        Does this recursively if has nested models.
    """
    model.supports_masking = True
    # TODO: Set default 'compute_mask(input, mask)' method of model here.
    #   See 'compute_mask' of keras.layers.Layer class versus 'compute_mask' of
    #   keras.models.Model class.
    level = kwargs.get("level", 0)  # only used for printing

    for layer in model.layers:
        if verbose and level is not None:
            print("".join(["\t"] * level), layer, flush=True)
        layer.supports_masking = True
        if issubclass(layer.__class__, tf.keras.Model):
            set_supports_masking(layer, verbose, level=level + 1)


def round(x, decimals=0):
    mult = tf.constant(10 ** decimals, dtype=x.dtype)
    rounded = tf.round(x * mult) / mult
    return rounded


def resize(image, min_side=800, max_side=1333):
    """
    image with rank 3 [h, w, c]
    """

    # if tf.equal(tf.rank(image), 3):
    #     h = tf.cast(tf.shape(image)[0], tf.float32)
    #     w = tf.cast(tf.shape(image)[1], tf.float32)
    # elif tf.equal(tf.rank(image), 4):
    #     h = tf.cast(tf.shape(image)[1], tf.float32)
    #     w = tf.cast(tf.shape(image)[2], tf.float32)
    # else:
    #     raise ValueError("Image must have rank 3 or 4. Found rank {}".format(tf.rank(image)))
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


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
def linear_sum_assignment(cost_matrix):
    nans = tf.math.is_nan(cost_matrix)
    if tf.reduce_any(nans):  # Convert nan to inf
        cost_matrix = tf.where(nans, tf.fill(tf.shape(cost_matrix), np.inf), cost_matrix)

    assignment = tf.py_function(func=linear_sum_assignment_scipy, inp=[cost_matrix], Tout=[tf.int32, tf.int32])
    assignment = tf.stack(assignment, axis=1)
    return assignment


@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.float32)])
def batch_linear_sum_assignment(cost_matrices):
    batch_size = cost_matrices.bounding_shape()[0]

    cost_matrix = cost_matrices[0].to_tensor()
    lsa = linear_sum_assignment(cost_matrix)
    for i in tf.range(1, batch_size):
        cost_matrix = cost_matrices[i].to_tensor()
        lsa_i = linear_sum_assignment(cost_matrix)
        lsa = tf.concat([lsa, lsa_i], axis=0)

    return lsa


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
def repeat_indices(repeats):
    """

    Example:
        >>> index_repeats = [1, 5, 2]
        >>> indices = repeat_indices(index_repeats)
        >>> indices
        [0, 1, 1, 1, 1, 1, 2, 2]

    """
    batch_size = tf.shape(repeats)[0]
    arr = tf.repeat(0, repeats[0])
    for i in tf.range(1, batch_size):
        arr = tf.concat([arr, tf.repeat(i, repeats[i])], axis=0)
    return arr
