import numpy as np

np.random.seed(42)
import tensorflow as tf

from chambers.layers.masking import DownsampleMasking, ReshapeWithMask
from chambers.layers.transformer import TransformerEncoder, TransformerDecoder
from chambers.layers.embedding import PositionalEmbedding2D
# from chambers.models.resnet import ResNet50Backbone
from models.backbone import ResNet50Backbone
from chambers.utils.tf import set_supports_masking


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


def DETR(input_shape, n_classes, n_query_embeds, embed_dim, num_heads, dim_feedforward, num_encoder_layers,
         num_decoder_layers, dropout_rate=0.1, return_decode_sequence=False, mask_value=None, name="detr"):
    inputs = tf.keras.layers.Input(shape=input_shape)

    if mask_value is not None:
        x_enc = tf.keras.layers.Masking(mask_value=mask_value)(inputs)
    else:
        x_enc = inputs

    # backbone = tf.keras.applications.ResNet50(input_shape=input_shape,
    #                                           include_top=False,
    #                                           weights="imagenet")
    backbone = ResNet50Backbone(name="backbone/0/body")
    # backbone = ResNet50Backbone(input_shape)
    x_enc = backbone(x_enc)

    x_enc = DownsampleMasking()(x_enc)

    proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=1, name='input_proj')
    x_enc = proj(x_enc)

    x_enc = PositionalEmbedding2D(embed_dim, normalize=True)(x_enc)
    x_enc = ReshapeWithMask([-1, embed_dim], [-1])(x_enc)  # (batch_size, h*w, embed_dim)

    if mask_value is not None:
        set_supports_masking(backbone, verbose=False)
        proj.supports_masking = True

    x = TransformerEncoder(embed_dim, num_heads, dim_feedforward, num_encoder_layers, dropout_rate, norm=False)(x_enc)
    x = TransformerDecoderDETR(n_query_embeds, embed_dim, num_heads, dim_feedforward, num_decoder_layers, dropout_rate,
                               norm=True, return_sequence=return_decode_sequence)(x)

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
# from chambers.optimizers import LearningRateMultiplier
#
# [v.name for v in model.get_layer("resnet50").variables]
#
#
# opt = LearningRateMultiplier(tf.keras.optimizers.Adam(), {"resnet50": 1e-5})
#
# mults = opt._get_params_multipliers(model.variables)
# mults.keys()
# len(mults[1e-5])

# [v.name for v in model.variables]

# %%


# x1 = np.random.normal(size=(batch_size, 544, 896, 3))
# x1 = np.pad(x1, [(0, 0), (0, 352), (0, 320), (0, 0)], mode="constant", constant_values=-1.)
# print(x1.shape)
#
# z = model(x1)
# print(z.shape)
# # print(z._keras_mask.shape)
