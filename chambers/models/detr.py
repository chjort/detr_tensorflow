import numpy as np

np.random.seed(42)
import tensorflow as tf

from chambers.layers.masking import DownsampleMasking, ReshapeWithMask
from chambers.layers.transformer import BaseTransformerDecoder, TransformerEncoderLayer, \
    BaseTransformerEncoder, TransformerDecoderLayer
from chambers.layers.embedding import PositionalEmbedding2D, LearnedEmbedding
from chambers.utils.tf import set_supports_masking


class TransformerEncoderLayerDETR(TransformerEncoderLayer):
    def call(self, inputs, training=None):
        v, pos_encoding = inputs

        q = k = v + pos_encoding

        attn_output = self.att([q, k, v])
        attn_output = self.dropout(attn_output, training=training)
        norm_output1 = self.layernorm1(v + attn_output)

        ffn_output = self.linear1(norm_output1)
        ffn_output = self.dropout1(ffn_output, training=training)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        norm_output2 = self.layernorm2(norm_output1 + ffn_output)

        return norm_output2


class TransformerDecoderLayerDETR(TransformerDecoderLayer):
    def call(self, inputs, training=None):
        v, enc_output, pos_enc, object_enc = inputs

        q = k = v + object_enc

        attn_output1 = self.attn1([q, k, v])
        attn_output1 = self.dropout1(attn_output1, training=training)
        norm_output1 = self.layernorm1(v + attn_output1)

        q = norm_output1 + object_enc
        k = enc_output + pos_enc

        attn_output2 = self.attn2([q, k, enc_output])
        attn_output2 = self.dropout2(attn_output2)
        norm_output2 = self.layernorm2(norm_output1 + attn_output2)

        ffn_output = self.linear1(norm_output1)
        ffn_output = self.dropout3(ffn_output, training=training)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout4(ffn_output, training=training)
        norm_output3 = self.layernorm3(norm_output2 + ffn_output)

        return norm_output3


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


class TransformerDecoderDETR(BaseTransformerDecoder):

    def __init__(self, n_object_queries, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1, norm=False,
                 return_sequence=False, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.n_object_queries = n_object_queries
        self.object_queries_layer = LearnedEmbedding(n_object_queries, embed_dim, add_to_input=False)

        layers = [TransformerDecoderLayerDETR(embed_dim, num_heads, ff_dim, dropout_rate)
                  for i in range(num_layers)]
        super(TransformerDecoderDETR, self).__init__(layers=layers, norm=norm, return_sequence=return_sequence,
                                                     **kwargs)

    def call(self, inputs, **kwargs):
        enc_output, pos_enc = inputs

        batch_size = tf.shape(enc_output)[0]
        x = tf.zeros([batch_size, self.n_object_queries, self.embed_dim])
        object_queries = self.object_queries_layer(x)

        decode_sequence = []
        for layer in self.layers:
            x = layer([x, enc_output, pos_enc, object_queries])
            decode_sequence.append(x)

        if self.norm:
            x = self._norm_layer(x)
            decode_sequence = [self._norm_layer(x) for x in decode_sequence]

        if self.return_sequence:
            x = tf.stack(decode_sequence, axis=0)
            x = tf.transpose(x, [1, 0, 2, 3])

        return x

    def get_config(self):
        config = {"n_object_queries": self.n_object_queries, "embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate, "num_layers": self.num_layers}
        base_config = super(TransformerDecoderDETR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def DETR(input_shape, n_classes, n_object_queries, embed_dim, num_heads, dim_feedforward, num_encoder_layers,
         num_decoder_layers, dropout_rate=0.1, return_decode_sequence=False, mask_value=None, name="detr"):
    inputs = tf.keras.layers.Input(shape=input_shape)

    if mask_value is not None:
        x_enc = tf.keras.layers.Masking(mask_value=mask_value)(inputs)
    else:
        x_enc = inputs

    backbone = tf.keras.applications.ResNet50(input_shape=input_shape,
                                              include_top=False,
                                              weights="imagenet")
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    x_enc = backbone(x_enc)
    x_enc = DownsampleMasking()(x_enc)

    proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=1, name='input_proj')
    x_enc = proj(x_enc)

    if mask_value is not None:
        set_supports_masking(backbone, verbose=False)
        proj.supports_masking = True

    pos_enc = PositionalEmbedding2D(embed_dim, normalize=True, add_to_input=False)(x_enc)
    pos_enc = tf.keras.layers.Reshape([-1, embed_dim])(pos_enc)  # (batch_size, h*w, embed_dim)
    x_enc = ReshapeWithMask([-1, embed_dim], [-1])(x_enc)  # (batch_size, h*w, embed_dim)

    enc_output = TransformerEncoderDETR(embed_dim, num_heads, dim_feedforward, num_encoder_layers, dropout_rate,
                                        norm=False)([x_enc, pos_enc])
    x = TransformerDecoderDETR(n_object_queries, embed_dim, num_heads, dim_feedforward, num_decoder_layers,
                               dropout_rate, norm=True, return_sequence=return_decode_sequence)([enc_output, pos_enc])

    n_classes = n_classes + 1  # Add 1 for the "Nothing" class
    x_class = tf.keras.layers.Dense(n_classes, name="class_embed")(x)
    x_box = tf.keras.Sequential([
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid")],
        name='bbox_embed')(x)

    x = tf.keras.layers.Concatenate(axis=-1)([x_class, x_box])

    model = tf.keras.models.Model(inputs, x, name=name)

    return model


def load_detr(path):
    if path.endswith(".h5") or path.endswith(".hdf5"):
        from tensorflow_addons.optimizers import AdamW
        from chambers.layers.masking import ReshapeWithMask, DownsampleMasking
        from chambers.layers.embedding import PositionalEmbedding2D
        from chambers.optimizers import LearningRateMultiplier
        from chambers.losses.hungarian import HungarianLoss
        detr = tf.keras.models.load_model(path,
                                          custom_objects={
                                              "DownsampleMasking": DownsampleMasking,
                                              "PositionalEmbedding2D": PositionalEmbedding2D,
                                              "ReshapeWithMask": ReshapeWithMask,
                                              "TransformerEncoderDETR": TransformerEncoderDETR,
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
# n_object_queries_ = 100
# batch_size = 2
# input_shape = (896, 1216, 3)
# # input_shape = (None, None, 3)
# return_sequence = False
#
# model = DETR(input_shape=input_shape,
#              n_classes=num_classes,
#              n_object_queries=n_object_queries_,
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
