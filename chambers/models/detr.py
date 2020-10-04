import tensorflow
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW

from chambers.layers.embedding import PositionalEmbedding2D
from chambers.layers.masking import DownsampleMasking, ReshapeWithMask
from chambers.layers.transformer import BaseDecoder, EncoderLayer, \
    BaseEncoder, DecoderLayer
from chambers.utils.tf import set_supports_masking


class EncoderLayerDETR(EncoderLayer):
    def call(self, inputs, mask=None, training=None):
        inputs, pos_encoding = inputs
        if mask is not None:
            mask = [mask[0], mask[0]]

        q = v = k = inputs
        q = q + pos_encoding
        k = k + pos_encoding

        attention = self.multi_head_attention([q, v, k], mask=mask)
        attention = self.dropout_attention(attention, training=training)
        x = self.add_attention([inputs, attention])
        x = self.layer_norm_attention(x)

        # Feed Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training=training)
        x = self.add_dense([x, dense])
        x = self.layer_norm_dense(x)

        return x


class DecoderLayerDETR(DecoderLayer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(DecoderLayerDETR, self).__init__(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                                               dropout_rate=dropout_rate, causal=False)

    def call(self, inputs, mask=None, training=None):
        x, x_encoder, pos_encoding, object_queries = inputs
        if mask is not None:
            mask = [None, mask[1]]

        q = x + object_queries
        v = x
        k = x + object_queries

        attention = self.multi_head_attention1([q, v, k], mask=None)
        attention = self.dropout_attention1(attention, training=training)
        x = self.add_attention1([x, attention])
        x = self.layer_norm_attention1(x)

        q2 = x + object_queries
        v2 = x_encoder
        k2 = x_encoder + pos_encoding

        attention = self.multi_head_attention2([q2, v2, k2], mask=mask)
        attention = self.dropout_attention2(attention, training=training)
        x = self.add_attention2([x, attention])
        x = self.layer_norm_attention2(x)

        # Feed Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training=training)
        x = self.add_dense([x, dense])
        x = self.layer_norm_dense(x)

        return x


class EncoderDETR(BaseEncoder):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1, norm=False, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        layers = [EncoderLayerDETR(embed_dim, num_heads, ff_dim, dropout_rate)
                  for i in range(num_layers)]
        super(EncoderDETR, self).__init__(layers=layers, norm=norm, **kwargs)

    def call(self, inputs, mask=None, **kwargs):
        x, pos_encoding = inputs

        for layer in self.layers:
            x = layer([x, pos_encoding], mask=mask)

        if self.norm:
            x = self.norm_layer(x)

        return x

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate,
                  "num_layers": self.num_layers}
        base_config = super(EncoderDETR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecoderDETR(BaseDecoder):

    def __init__(self, n_object_queries, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1, norm=False,
                 return_sequence=False, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.n_object_queries = n_object_queries

        layers = [DecoderLayerDETR(embed_dim, num_heads, ff_dim, dropout_rate)
                  for i in range(num_layers)]
        super(DecoderDETR, self).__init__(layers=layers, norm=norm, return_sequence=return_sequence,
                                          **kwargs)

    def build(self, input_shape):
        self.object_queries = self.add_weight("object_queries",
                                              shape=(1, self.n_object_queries, self.embed_dim),
                                              initializer="normal",
                                              trainable=True,
                                              )

    def call(self, inputs, mask=None, **kwargs):
        pos_encoding, x_encoder = inputs
        if mask is not None:
            mask = [None, mask[1]]

        batch_size = tf.shape(x_encoder)[0]
        x = tf.zeros([batch_size, self.n_object_queries, self.embed_dim])

        decode_sequence = []
        for layer in self.layers:
            x = layer([x, x_encoder, pos_encoding, self.object_queries], mask=mask)
            decode_sequence.append(x)

        if self.norm:
            decode_sequence = [self.norm_layer(x) for x in decode_sequence]

        if self.return_sequence:
            x = tf.stack(decode_sequence, axis=0)
            x = tf.transpose(x, [1, 0, 2, 3])
        else:
            x = decode_sequence[-1]

        return x

    def get_config(self):
        config = {"n_object_queries": self.n_object_queries, "embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate, "num_layers": self.num_layers}
        base_config = super(DecoderDETR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def DETR(input_shape, n_classes, n_object_queries, embed_dim, num_heads, dim_feedforward, num_encoder_layers,
         num_decoder_layers, dropout_rate=0.1, return_decode_sequence=False, mask_value=None, name="detr"):
    inputs = tf.keras.layers.Input(shape=input_shape)

    if mask_value is not None:
        x_enc = tf.keras.layers.Masking(mask_value=mask_value)(inputs)
    else:
        x_enc = inputs

    with tensorflow.python.keras.backend.get_graph().as_default(), tf.name_scope("resnet50"):
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
    pos_enc = tf.keras.layers.Reshape([-1, embed_dim], name="positional_embedding_sequence")(
        pos_enc)  # (batch_size, h*w, embed_dim)
    x_enc = ReshapeWithMask([-1, embed_dim], [-1], name="image_features_sequence")(
        x_enc)  # (batch_size, h*w, embed_dim)

    enc_output = EncoderDETR(embed_dim, num_heads, dim_feedforward, num_encoder_layers, dropout_rate,
                             norm=False)([x_enc, pos_enc])
    x = DecoderDETR(n_object_queries, embed_dim, num_heads, dim_feedforward, num_decoder_layers,
                    dropout_rate, norm=True, return_sequence=return_decode_sequence)([pos_enc, enc_output])

    n_classes = n_classes + 1  # Add 1 for the "Nothing" class
    x_class = tf.keras.layers.Dense(n_classes, name="class_embed")(x)
    x_box = tf.keras.Sequential([
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(embed_dim, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid")],
        name='bbox_embed')(x)

    x = tf.keras.layers.Concatenate(axis=-1)([x_class, x_box])

    # x = enc_output
    model = tf.keras.models.Model(inputs, x, name=name)

    return model


def load_detr(path, compile=True):
    if path.endswith(".h5") or path.endswith(".hdf5"):
        from chambers.layers.masking import ReshapeWithMask, DownsampleMasking
        from chambers.layers.embedding import PositionalEmbedding2D
        from chambers.optimizers import LearningRateMultiplier
        from chambers.losses.hungarian import HungarianLoss
        detr = tf.keras.models.load_model(path,
                                          custom_objects={
                                              "DownsampleMasking": DownsampleMasking,
                                              "PositionalEmbedding2D": PositionalEmbedding2D,
                                              "ReshapeWithMask": ReshapeWithMask,
                                              "EncoderDETR": EncoderDETR,
                                              "DecoderDETR": DecoderDETR,
                                              "LearningRateMultiplier": LearningRateMultiplier,
                                              "Addons>AdamW": AdamW,
                                              "HungarianLoss": HungarianLoss
                                          },
                                          compile=compile)
    else:
        from chambers.losses.hungarian import HungarianLoss
        detr = tf.keras.models.load_model(path,
                                          custom_objects={
                                              "HungarianLoss": HungarianLoss
                                          },
                                          compile=compile)

    return detr


def post_process(y_pred, min_prob=None):
    boxes = y_pred[..., :4]
    logits = y_pred[..., 4:]

    softmax = tf.nn.softmax(logits, axis=-1)[..., :-1]
    labels = tf.argmax(softmax, axis=-1)
    probs = tf.reduce_max(softmax, axis=-1)

    boxes = boxes.numpy()
    labels = labels.numpy()
    probs = probs.numpy()

    if min_prob is not None:
        min_prob = float(min_prob)
        keep_mask = probs > min_prob
        boxes = boxes[keep_mask]
        labels = labels[keep_mask]
        probs = probs[keep_mask]

    return boxes, labels, probs


tf.keras.utils.get_custom_objects().update({
    "EncoderLayerDETR": EncoderLayerDETR,
    "DecoderLayerDETR": DecoderLayerDETR,
    "EncoderDETR": EncoderDETR,
    "DecoderDETR": DecoderDETR,
    "Addons>AdamW": AdamW,
})
