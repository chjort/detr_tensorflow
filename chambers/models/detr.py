import numpy as np

np.random.seed(42)
import tensorflow as tf

from chambers.layers.masking import DownsampleMasking
from chambers.layers.transformer import TransformerEncoder, TransformerDecoder
from chambers.layers.embedding import PositionalEmbedding2D
from chambers.models.resnet import ResNet50Backbone
from chambers.utils.tf import set_supports_masking

class TransformerDecoderDETR(TransformerDecoder):
    """
    A standard transformer decoder with its target input fixed to be query embeddings.
    Therefore, this decoder only takes input from an encoder.

    """

    def __init__(self, n_query_embeds, embed_dim, num_heads, dim_feedforward, num_layers, dropout_rate=0.1, norm=False,
                 return_sequence=False, **kwargs):
        super(TransformerDecoderDETR, self).__init__(embed_dim, num_heads, dim_feedforward, num_layers, dropout_rate,
                                                     norm, return_sequence, **kwargs)
        self.n_query_embeds = n_query_embeds
        self.query_embeddings = tf.keras.layers.Embedding(n_query_embeds, embed_dim)

    def call(self, inputs, **kwargs):
        # `inputs` are the encoder output

        batch_size = tf.shape(inputs)[0]
        x = tf.tile(tf.expand_dims(tf.range(self.n_query_embeds), 0), [batch_size, 1])
        x = self.query_embeddings(x)

        x = super().call([x, inputs], **kwargs)

        return x


def DETR(input_shape, n_classes, n_query_embeds, embed_dim, num_heads, dim_feedforward, num_encoder_layers,
         num_decoder_layers, dropout_rate=0.1, return_decode_sequence=False, mask_value=None, name="detr"):
    inputs = tf.keras.layers.Input(shape=input_shape)

    if mask_value is not None:
        x_enc = tf.keras.layers.Masking(mask_value=mask_value)(inputs)
    else:
        x_enc = inputs

    x_enc = ResNet50Backbone(input_shape)(x_enc)
    x_enc = DownsampleMasking()(x_enc)
    x_enc = tf.keras.layers.Conv2D(embed_dim, kernel_size=1, name='input_proj')(x_enc)
    x_enc = PositionalEmbedding2D(embed_dim, normalize=True)(x_enc)
    x_enc = tf.keras.layers.Reshape([-1, embed_dim])(x_enc)  # (batch_size, h*w, embed_dim)

    # x = TransformerEncoder(embed_dim, num_heads, dim_feedforward, num_encoder_layers, dropout_rate, norm=False)(x_enc)
    # x = TransformerDecoderDETR(n_query_embeds, embed_dim, num_heads, dim_feedforward, num_decoder_layers, dropout_rate,
    #                            norm=True, return_sequence=return_decode_sequence)(x)
    #
    # n_classes = n_classes + 1  # Add 1 for the "Nothing" class
    # x_class = tf.keras.layers.Dense(n_classes, name="class_embed")(x)
    # x_box = tf.keras.Sequential([
    #     tf.keras.layers.Dense(embed_dim, activation="relu"),
    #     tf.keras.layers.Dense(embed_dim, activation="relu"),
    #     tf.keras.layers.Dense(4, activation="sigmoid")],
    #     name='bbox_embed')(x)
    #
    # x = tf.keras.layers.Concatenate(axis=-1)([x_class, x_box])

    x = x_enc
    model = tf.keras.models.Model(inputs, x, name=name)

    if mask_value is not None:
        set_supports_masking(model, verbose=False)

    return model


# %%
num_classes = 91
embed_dim = 256
n_query_embeds_ = 100
batch_size = 2
input_shape = (896, 1216, 3)
# input_shape = (None, None, 3)
return_sequence = False

model = DETR(input_shape=input_shape,
             n_classes=num_classes,
             n_query_embeds=n_query_embeds_,
             embed_dim=embed_dim,
             num_heads=8,
             dim_feedforward=2048,
             num_encoder_layers=6,
             num_decoder_layers=6,
             dropout_rate=0.1,
             return_decode_sequence=return_sequence,
             mask_value=-1.
             )

model.summary()

# %%
x1 = np.random.normal(size=(batch_size, 544, 896, 3))
x1 = np.pad(x1, [(0, 0), (0, 352), (0, 320), (0, 0)], mode="constant", constant_values=-1.)
print(x1.shape)

z = model(x1)
print(z.shape)
print(z._keras_mask.shape)
