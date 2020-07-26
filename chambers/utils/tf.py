import tensorflow as tf


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
