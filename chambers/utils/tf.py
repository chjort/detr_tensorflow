import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as linear_sum_assignment_scipy


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


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
def linear_sum_assignment(cost_matrix):
    nans = tf.math.is_nan(cost_matrix)
    if tf.reduce_any(nans):  # Convert nan to inf
        cost_matrix = tf.where(nans, tf.fill(tf.shape(cost_matrix), np.inf), cost_matrix)

    assignment = tf.py_function(func=linear_sum_assignment_scipy, inp=[cost_matrix], Tout=[tf.int32, tf.int32])
    assignment = tf.stack(assignment, axis=1)
    return assignment


@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.float32)])
def linear_sum_assignment_ragged(cost_matrix):
    cost_matrix = cost_matrix.to_tensor()
    assignment = linear_sum_assignment(cost_matrix)
    assignment = tf.RaggedTensor.from_tensor(assignment)
    return assignment


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
def batch_linear_sum_assignment(cost_matrices):
    res = tf.map_fn(linear_sum_assignment, elems=cost_matrices, fn_output_signature=tf.int32)
    res = tf.reshape(res, [-1, 2])
    return res


@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.float32)])
def batch_linear_sum_assignment_ragged(cost_matrices):
    res = tf.map_fn(linear_sum_assignment_ragged, elems=cost_matrices,
                    fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.int32))
    res = tf.reshape(res.flat_values, [-1, 2])
    return res


def lsa_to_batch_indices(lsa_indices, batch_mask):
    sizes = tf.reduce_sum(tf.cast(batch_mask, tf.int32), axis=1)
    row_idx = repeat_indices(sizes)
    row_idx = tf.tile(tf.expand_dims(row_idx, -1), [1, 2])
    indcs = tf.stack([row_idx, lsa_indices], axis=0)

    prediction_idx = tf.transpose(indcs[:, :, 0])
    target_idx = tf.transpose(indcs[:, :, 1])
    return prediction_idx, target_idx