import tensorflow as tf


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


def pairwise_subtract(a, b):
    tf.debugging.assert_rank(b, tf.rank(a), "Tensor 'b' must have same rank as tensor 'a'")

    a_dim = tf.rank(a) - 1
    b_dim = tf.rank(b) - 2
    a = tf.expand_dims(a, axis=a_dim)
    b = tf.expand_dims(b, axis=b_dim)
    return a - b


def pairwise_l1(a, b):
    """ L1 Distance (Manhattan distance) """
    x = pairwise_subtract(a, b)
    x = tf.abs(x)
    x = tf.reduce_sum(x, -1)
    return x


def pairwise_l2(a, b):
    """ L2 Distance (Euclidean distance) """
    x = pairwise_subtract(a, b)
    x = tf.square(x)
    x = tf.reduce_sum(x, -1)
    x = tf.sqrt(x)
    return x


def round(x, decimals=0):
    mult = tf.constant(10 ** decimals, dtype=x.dtype)
    rounded = tf.round(x * mult) / mult
    return rounded

# @tf.function
# def tf_linear_sum_assignment(cost_matrix):
#     assignment = tf.py_function(func=linear_sum_assignment, inp=[cost_matrix], Tout=[tf.int64, tf.int64])
#     return assignment
def resize(image, min_side=800, max_side=1333):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cur_min_side = tf.minimum(w, h)
    cur_max_side = tf.maximum(w, h)

    min_side = tf.cast(min_side, tf.float32)
    max_side = tf.cast(max_side, tf.float32)
    scale = tf.minimum(max_side / cur_max_side,
                       min_side / cur_min_side)
    nh = tf.cast(scale * h, tf.int32)
    nw = tf.cast(scale * w, tf.int32)

    image = tf.image.resize(image, (nh, nw))
    return image