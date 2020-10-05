import time

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from chambers.utils.tf import linear_sum_assignment_scipy, linear_sum_assignment

parallel = Parallel(n_jobs=-1)


def timeit(fn, n=1, *args, **kwargs):
    times = []
    fn(*args, **kwargs)  # warm up
    for i in range(n):
        st = time.time()
        fn(*args, **kwargs)
        end_t = time.time() - st
        times.append(end_t)
    return np.mean(times)


def batch_lsa_sp_mp(cost_matrices):
    res = parallel(
        delayed(linear_sum_assignment_scipy)(cost_mat) for cost_mat in cost_matrices
    )


def batch_lsa_sp(cost_matrices):
    res = [linear_sum_assignment_scipy(cost_mat) for cost_mat in cost_matrices]


@tf.function
def batch_lsa_tf(cost_matrices):
    res = tf.map_fn(linear_sum_assignment, elems=cost_matrices, fn_output_signature=tf.int32)
    # res = tf.vectorized_map(linear_sum_assignment, elems=cost_matrices)
    res = tf.reshape(res, [-1, 2])
    return res


@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.float32)])
def linear_sum_assignment_rag(cost_matrix):
    cost_matrix = cost_matrix.to_tensor()
    assignment = linear_sum_assignment(cost_matrix)
    assignment = tf.RaggedTensor.from_tensor(assignment)
    return assignment


@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.float32)])
def batch_lsa_tf_rag(cost_matrices):
    res = tf.map_fn(linear_sum_assignment_rag, elems=cost_matrices,
                    fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.int32))
    res = tf.reshape(res.flat_values, [-1, 2])
    return res


# %%
bz = 200
costs = np.random.normal(size=(bz, 300, 300)).astype(np.float32)
costs_tf = tf.stack(costs)

cost_r = [np.random.normal(size=(300, 300)).astype(np.float32) for _ in range(bz - 1)]
cost_r.append(np.random.normal(size=(300, 299)).astype(np.float32))
costs_tfr = tf.ragged.constant(cost_r)
costs_tfr.shape

mean1 = timeit(batch_lsa_sp, 5, costs)
mean2 = timeit(batch_lsa_tf, 5, costs_tf)
mean3 = timeit(batch_lsa_tf_rag, 5, costs_tfr)

print(mean1)
print(mean2)
print(mean3)
# print(1 - (mean2 / mean1), "percent faster")
