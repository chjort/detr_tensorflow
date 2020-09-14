import tensorflow as tf


class InputMeanMaxMinMetricLayer(tf.keras.layers.Layer):
    def __init__(self, prefix=None):
        super(InputMeanMaxMinMetricLayer, self).__init__()
        self.prefix = prefix
        if self.prefix is None:
            self.prefix = ""

    def call(self, inputs, **kwargs):
        self.add_metric(tf.reduce_mean(inputs, axis=(1, 2, 3)),
                        aggregation="mean",
                        name=self.prefix + "mean")
        self.add_metric(tf.reduce_max(inputs, axis=(1, 2, 3)),
                        aggregation="mean",
                        name=self.prefix + "max")
        self.add_metric(tf.reduce_min(inputs, axis=(1, 2, 3)),
                        aggregation="mean",
                        name=self.prefix + "min")
        return inputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"prefix": self.prefix}
        base_config = super(InputMeanMaxMinMetricLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
