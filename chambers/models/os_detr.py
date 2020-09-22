import numpy as np

np.random.seed(42)
import tensorflow as tf

from chambers.layers.embedding import PositionalEmbedding2D
from chambers.models.resnet import ResNet50Backbone
from chambers.models.transformer import Transformer


def OSDETR(input_shape, embed_dim, num_heads, dim_feedforward, num_encoder_layers,
           num_decoder_layers, dropout_rate=0.1, return_decode_sequence=False, name="osdetr"):
    inputs = tf.keras.layers.Input(shape=input_shape, name="target")
    inputs2 = tf.keras.layers.Input(shape=input_shape, name="query")

    backbone = ResNet50Backbone(input_shape)

    # TODO: Should encoder input be target image, and decoder input query image? or vice versa?
    x_enc = backbone(inputs)
    x_enc = tf.keras.layers.Conv2D(embed_dim, kernel_size=1, name='target_proj')(x_enc)
    x_enc = PositionalEmbedding2D(embed_dim, normalize=True, name="target_pos_emb")(x_enc)
    x_enc = tf.keras.layers.Reshape([-1, embed_dim], name="target_features")(x_enc)  # (batch_size, h*w, embed_dim)

    x_dec = backbone(inputs2)
    x_dec = tf.keras.layers.Conv2D(embed_dim, kernel_size=1, name='query_proj')(x_dec)
    x_dec = PositionalEmbedding2D(embed_dim, normalize=True, name="query_pos_emb")(x_dec)
    x_dec = tf.keras.layers.Reshape([-1, embed_dim], name="query_features")(x_dec)  # (batch_size, h*w, embed_dim)
    # TODO: make decoder input a fixed sized sequence. Example (batch_size, 100, embed_dim)

    transformer = Transformer(input_shape=[(x_enc.shape[1], embed_dim), (x_dec.shape[1], embed_dim)],
                              num_heads=num_heads,
                              dim_feedforward=dim_feedforward,
                              num_encoder_layers=num_encoder_layers,
                              num_decoder_layers=num_decoder_layers,
                              dropout_rate=dropout_rate,
                              return_decode_sequence=return_decode_sequence
                              )
    x = transformer([x_enc, x_dec])

    x_match = tf.keras.layers.Dense(1, activation="sigmoid", name="match_head")(x)
    x_box = tf.keras.Sequential([
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid")],
        name='bbox_head')(x)

    x = tf.keras.layers.Concatenate(axis=-1)([x_match, x_box])

    model = tf.keras.models.Model([inputs, inputs2], x, name=name)
    return model

# %%
# num_classes = 91
# embed_dim = 256
# n_object_queries_ = 100
# batch_size = 2
# input_shape = (896, 1216, 3)
# # input_shape = (None, None, 3)
# return_sequence = False
#
# model = OSDETR(input_shape=input_shape,
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

