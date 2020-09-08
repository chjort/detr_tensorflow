import tensorflow as tf


class DownsampleMasking(tf.keras.layers.Layer):

    def call(self, inputs, mask=None, **kwargs):
        # NOTE: 'call' method MUST modify input, in order for new mask to be computed.

        if mask is not None:
            boolean_mask = tf.expand_dims(self.compute_mask(inputs, inputs._keras_mask), -1)
            inputs = inputs * tf.cast(boolean_mask, inputs.dtype)

        return inputs

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None

        masks = tf.cast(mask, inputs.dtype)
        masks = tf.expand_dims(masks, -1)
        # The existing tf.image.resize with method='nearest'
        # does not expose the half_pixel_centers option in TF 2.2.0
        # The original Pytorch F.interpolate uses it like this
        if tf.keras.backend.image_data_format() == "channels_last":
            hw = tf.shape(inputs)[1:3]
        else:
            hw = tf.shape(inputs)[2:4]
        masks = tf.compat.v1.image.resize_nearest_neighbor(
            masks, hw, align_corners=False, half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        return masks


class ReshapeWithMask(tf.keras.layers.Reshape):
    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            # TODO:
            # mask = tf.reshape(mask, (tf.shape(inputs)[0],) + self.target_shape[:-1])
            pass

        return mask
