import pickle

import numpy as np

np.random.seed(42)
import tensorflow as tf

from chambers.layers.masking import DownsampleMasking, ReshapeWithMask
from chambers.layers.transformer import TransformerDecoder, TransformerEncoderLayer, \
    BaseTransformerEncoder, TransformerEncoder
from chambers.layers.embedding import PositionalEmbedding2D
from chambers.models.resnet import ResNet50Backbone
from chambers.utils.tf import set_supports_masking


class TransformerEncoderLayerDETR(TransformerEncoderLayer):
    def call(self, inputs, training=None):
        v, pos_encoding = inputs

        q = v + pos_encoding
        k = v + pos_encoding

        attn_output = self.att([q, k, v])
        attn_output = self.dropout(attn_output, training=training)
        norm_output1 = self.layernorm1(v + attn_output)

        ffn_output = self.linear1(norm_output1)
        ffn_output = self.dropout1(ffn_output, training=training)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        norm_output2 = self.layernorm2(norm_output1 + ffn_output)

        return norm_output2


class TransformerEncoderDETR(BaseTransformerEncoder):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1, norm=False, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        layers = [TransformerEncoderLayerDETR(embed_dim, num_heads, ff_dim, dropout_rate)
                  for i in range(num_layers)]
        super(TransformerEncoderDETR, self).__init__(layers=layers, norm=norm, **kwargs)

    def call(self, inputs, **kwargs):
        x, pos_encoding = inputs

        for layer in self.layers:
            x = layer([x, pos_encoding])

        if self.norm:
            x = self._norm_layer(x)

        return x

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate,
                  "num_layers": self.num_layers}
        base_config = super(TransformerEncoderDETR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TransformerDecoderDETR(TransformerDecoder):
    """
    A standard transformer decoder with its target input fixed to be query embeddings.
    Therefore, this decoder only takes input from an encoder.

    """

    def __init__(self, n_query_embeds, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1, norm=False,
                 return_sequence=False, **kwargs):
        super(TransformerDecoderDETR, self).__init__(embed_dim, num_heads, ff_dim, num_layers, dropout_rate,
                                                     norm, return_sequence, **kwargs)
        self.n_query_embeds = n_query_embeds
        self.query_embeddings = self.add_weight("query_embeddings",
                                                shape=[1, n_query_embeds, embed_dim],
                                                initializer="zeros")

    def call(self, inputs, **kwargs):
        # `inputs` are the encoder output

        batch_size = tf.shape(inputs)[0]
        x = tf.tile(self.query_embeddings, [batch_size, 1, 1])

        x = super().call([x, inputs], **kwargs)

        return x

    def get_config(self):
        config = {"n_query_embeds": self.n_query_embeds}
        base_config = super(TransformerDecoderDETR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PrintLayer(tf.keras.layers.Layer):
    def __init__(self, prefix=None):
        super(PrintLayer, self).__init__()
        self.prefix = prefix

    def call(self, inputs, **kwargs):
        if self.prefix is not None:
            # tf.print(self.prefix, tf.reduce_any(tf.math.is_nan(inputs)), tf.reduce_max(inputs), tf.reduce_min(inputs))
            tf.print(self.prefix, tf.reduce_mean(inputs), tf.reduce_max(inputs), tf.reduce_min(inputs))
        else:
            # tf.print(tf.reduce_any(tf.math.is_nan(inputs)), tf.reduce_max(inputs), tf.reduce_min(inputs))
            tf.print(tf.reduce_mean(inputs), tf.reduce_max(inputs), tf.reduce_min(inputs))
        return inputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"prefix": self.prefix}
        base_config = super(PrintLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PrintConv2D(tf.keras.layers.Conv2D):

    def call(self, inputs):
        tf.print("conv2d_1", tf.reduce_any(tf.math.is_nan(inputs)), tf.reduce_max(inputs), tf.reduce_min(inputs))
        x = super(PrintConv2D, self).call(inputs)
        tf.print("conv2d_2", tf.reduce_any(tf.math.is_nan(x)))

        return x


def DETR(input_shape, n_classes, n_query_embeds, embed_dim, num_heads, dim_feedforward, num_encoder_layers,
         num_decoder_layers, dropout_rate=0.1, return_decode_sequence=False, mask_value=None, name="detr",
         backbone_weights=None):
    inputs = tf.keras.layers.Input(shape=input_shape)

    if mask_value is not None:
        x_enc = tf.keras.layers.Masking(mask_value=mask_value)(inputs)
    else:
        x_enc = inputs

    x_enc = PrintLayer("\nbackbone_1")(x_enc)
    backbone = ResNet50Backbone(input_shape, name="backbone/0/body")
    x_enc = backbone(x_enc)
    x_enc = PrintLayer("backbone_2")(x_enc)
    x_enc = DownsampleMasking()(x_enc)

    proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=1, name='input_proj')
    # proj = PrintConv2D(embed_dim, kernel_size=1, name='input_proj')
    x_enc = proj(x_enc)

    if mask_value is not None:
        set_supports_masking(backbone, verbose=False)
        proj.supports_masking = True

    x_enc = PositionalEmbedding2D(embed_dim, normalize=True)(x_enc)
    x_enc = ReshapeWithMask([-1, embed_dim], [-1])(x_enc)  # (batch_size, h*w, embed_dim)
    # pos_enc = PositionalEmbedding2D(embed_dim, normalize=True, add_to_input=False)(x_enc)
    # pos_enc = tf.keras.layers.Reshape([-1, embed_dim])(pos_enc)  # (batch_size, h*w, embed_dim)
    # x_enc = ReshapeWithMask([-1, embed_dim], [-1])(x_enc)  # (batch_size, h*w, embed_dim)

    enc_output = TransformerEncoder(embed_dim, num_heads, dim_feedforward, num_encoder_layers, dropout_rate,
                                    norm=False)(x_enc)
    # enc_output = TransformerEncoderDETR(embed_dim, num_heads, dim_feedforward, num_encoder_layers, dropout_rate,
    #                                     norm=False)([x_enc, pos_enc])
    x = TransformerDecoderDETR(n_query_embeds, embed_dim, num_heads, dim_feedforward, num_decoder_layers, dropout_rate,
                               norm=True, return_sequence=return_decode_sequence)(enc_output)

    n_classes = n_classes + 1  # Add 1 for the "Nothing" class
    x_class = tf.keras.layers.Dense(n_classes, name="class_embed")(x)
    x_box = tf.keras.Sequential([
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid")],
        name='bbox_embed')(x)

    x = tf.keras.layers.Concatenate(axis=-1)([x_class, x_box])

    if backbone_weights is not None:
        with open(backbone_weights, 'rb') as f:
            detr_weights = pickle.load(f)

        for var in backbone.variables:
            print(var.name)
            var = var.assign(detr_weights[var.name[:-2]])

    model = tf.keras.models.Model(inputs, x, name=name)

    return model


def load_detr(path):
    if path.endswith(".h5") or path.endswith(".hdf5"):
        from tensorflow_addons.optimizers import AdamW
        from chambers.layers.batch_norm import FrozenBatchNorm2D
        from chambers.layers.masking import ReshapeWithMask, DownsampleMasking
        from chambers.layers.embedding import PositionalEmbedding2D
        from chambers.layers.transformer import TransformerEncoder
        from chambers.models.detr import TransformerDecoderDETR
        from chambers.optimizers import LearningRateMultiplier
        from chambers.losses.hungarian import HungarianLoss
        tf.keras.models.Model.save()
        detr = tf.keras.models.load_model(path,
                                          custom_objects={
                                              "FrozenBatchNorm2D": FrozenBatchNorm2D,
                                              "DownsampleMasking": DownsampleMasking,
                                              "PositionalEmbedding2D": PositionalEmbedding2D,
                                              "ReshapeWithMask": ReshapeWithMask,
                                              "TransformerEncoder": TransformerEncoder,
                                              "TransformerDecoderDETR": TransformerDecoderDETR,
                                              "LearningRateMultiplier": LearningRateMultiplier,
                                              "Addons>AdamW": AdamW,
                                              "HungarianLoss": HungarianLoss
                                          })
    else:
        from chambers.losses.hungarian import HungarianLoss
        detr = tf.keras.models.load_model(path,
                                          custom_objects={
                                              "HungarianLoss": HungarianLoss
                                          })

    return detr

# %%
# num_classes = 91
# embed_dim = 256
# n_query_embeds_ = 100
# batch_size = 2
# input_shape = (896, 1216, 3)
# # input_shape = (None, None, 3)
# return_sequence = False
#
# model = DETR(input_shape=input_shape,
#              n_classes=num_classes,
#              n_query_embeds=n_query_embeds_,
#              embed_dim=embed_dim,
#              num_heads=8,
#              dim_feedforward=2048,
#              num_encoder_layers=6,
#              num_decoder_layers=6,
#              dropout_rate=0.1,
#              return_decode_sequence=return_sequence,
#              mask_value=-1.
#              )
# model.summary()

# %%
# x1 = np.random.normal(size=(batch_size, 544, 896, 3))
# x1 = np.pad(x1, [(0, 0), (0, 352), (0, 320), (0, 0)], mode="constant", constant_values=-1.)
# print(x1.shape)
#
# z = model(x1)
# print(z.shape)
# # print(z._keras_mask.shape)
