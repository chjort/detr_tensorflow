import numpy as np
np.random.seed(42)
import tensorflow as tf

from chambers.layers.embedding import PositionalEmbedding2D
from chambers.layers.shaping import Transpose
from chambers.models.resnet import ResNet50Backbone
from chambers.models.transformer import Transformer


# class TransformerDecoderDETR(TransformerDecoder):
#     def call(self, inputs, **kwargs):
#         batch_size = tf.shape(inputs)[0]
#         x = tf.tile(tf.expand_dims(tf.range(self.output_len), 0), [batch_size, 1])
#
#         x = super().call([x, inputs], **kwargs)
#
#         return x


# def TransformerDETR(input_shape, output_len, num_heads, dim_feedforward,
#                     num_encoder_layers, num_decoder_layers,
#                     dropout_rate=0.1, return_decode_sequence=False):
#     inputs = tf.keras.layers.Input(shape=input_shape)
#     embed_dim = input_shape[1]
#
#     x = inputs
#     x = TransformerEncoder(embed_dim, num_heads, dim_feedforward, num_encoder_layers, dropout_rate)(x)
#     x = TransformerDecoderDETR(output_len, embed_dim, num_heads, dim_feedforward, num_decoder_layers, dropout_rate,
#                                return_sequence=return_decode_sequence)(x)
#
#     model = tf.keras.models.Model(inputs, x)
#
#     return model

def DETR(input_shape, n_classes, n_query_embeds, embed_dim, num_heads, dim_feedforward, num_encoder_layers,
         num_decoder_layers, dropout_rate=0.1, return_decode_sequence=False, name="detr"):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x_enc = ResNet50Backbone(input_shape)(inputs)
    x_enc = tf.keras.layers.Conv2D(embed_dim, kernel_size=1, name='input_proj')(x_enc)
    x_enc = PositionalEmbedding2D(embed_dim, normalize=True)(x_enc)
    x_enc = tf.keras.layers.Reshape([-1, embed_dim])(x_enc)  # (batch_size, h*w, embed_dim)

    # x_dec = tf.tile(tf.expand_dims(tf.range(x_dec_len), 0), [batch_size, 1])
    # x_dec = tf.keras.layers.Embedding(x_dec_len, embed_dim)(x_dec)  # (2, 100, 256)
    x_dec = tf.keras.layers.Input(shape=(n_query_embeds, embed_dim))

    transformer = Transformer(input_shape=[(x_enc.shape[1], embed_dim), (n_query_embeds, embed_dim)],
                              num_heads=num_heads,
                              dim_feedforward=dim_feedforward,
                              num_encoder_layers=num_encoder_layers,
                              num_decoder_layers=num_decoder_layers,
                              dropout_rate=dropout_rate,
                              return_decode_sequence=return_decode_sequence
                              )
    x = transformer([x_enc, x_dec])

    n_classes = n_classes + 1  # Add 1 for the "Nothing" class
    x_class = tf.keras.layers.Dense(n_classes, name="class_embed")(x)
    x_box = tf.keras.Sequential([
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid")],
        name='bbox_embed')(x)

    # if return_decode_sequence:
    #     x_class = Transpose([1, 0, 2, 3])(x_class)
    #     x_box = Transpose([1, 0, 2, 3])(x_box)

    x = tf.keras.layers.Concatenate(axis=-1)([x_class, x_box])

    model = tf.keras.models.Model([inputs, x_dec], x, name=name)
    return model


# %%
num_classes = 91
embed_dim = 256
n_query_embeds_ = 100
# n_query_embeds_ = None
batch_size = 2
input_shape = (896, 1216, 3)
# input_shape = (None, None, 3)
return_sequence = True

model = DETR(input_shape, num_classes, n_query_embeds_, embed_dim, num_heads=8, dim_feedforward=2048,
             num_encoder_layers=6, num_decoder_layers=6, dropout_rate=0.1, return_decode_sequence=return_sequence)

model.summary()

# %%
x1 = np.random.normal(size=(batch_size, 896, 1216, 3))
x2 = np.random.normal(size=(batch_size, 100, embed_dim))
print(x1.shape, x2.shape)

z = model([x1, x2])
print(z.shape)

# TODO
""" DO THIS
transformer (Model)             (None, 6, 100, 256)  17363456    reshape[0][0]                    
                                                                 input_3[0][0]                    
__________________________________________________________________________________________________
class_embed (Dense)             (None, 6, 100, 92)   23644       transformer[1][0]                
__________________________________________________________________________________________________
bbox_embed (Sequential)         (None, 6, 100, 4)    132612      transformer[1][0]                
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 6, 100, 96)   0           class_embed[0][0]                
                                                                 bbox_embed[0][0]                 
==================================================================================================
Total params: 41,605,408
Trainable params: 41,499,168
Non-trainable params: 106,240
"""

""" OR THIS??
transformer (Model)             (6, None, 100, 256)  17363456    reshape[0][0]                    
                                                                 input_3[0][0]                    
__________________________________________________________________________________________________
class_embed (Dense)             (6, None, 100, 92)   23644       transformer[1][0]                
__________________________________________________________________________________________________
bbox_embed (Sequential)         (6, None, 100, 4)    132612      transformer[1][0]                
__________________________________________________________________________________________________
transpose (Transpose)           (None, 6, 100, 92)   0           class_embed[0][0]                
__________________________________________________________________________________________________
transpose_1 (Transpose)         (None, 6, 100, 4)    0           bbox_embed[0][0]                 
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 6, 100, 96)   0           transpose[0][0]                  
                                                                 transpose_1[0][0]                
==================================================================================================
Total params: 41,605,408
Trainable params: 41,499,168
Non-trainable params: 106,240
__________________________________________________________________________________________________

"""