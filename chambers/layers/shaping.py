import tensorflow as tf


class Transpose(tf.keras.layers.Layer):
    def __init__(self, permute=None, conjugate=False, **kwargs):
        super().__init__(**kwargs)
        self.permute = permute
        self.conjugate = conjugate

    def call(self, inputs, **kwargs):
        return tf.transpose(inputs, perm=self.permute, conjugate=self.conjugate)
