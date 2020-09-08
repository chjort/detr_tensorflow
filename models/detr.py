import pickle

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU

from chambers.layers import DownsampleMasking
from chambers.utils.boxes import box_cxcywh_to_xyxy
from chambers.utils.tf import set_supports_masking
from .backbone import ResNet50Backbone
from .custom_layers import Linear
from .position_embeddings import PositionEmbeddingSine
from .transformer import Transformer


class DETR(tf.keras.Model):
    def __init__(self, num_classes=91, num_queries=100,
                 mask_value=-1.,
                 backbone=None,
                 pos_encoder=None,
                 transformer=None,
                 return_decode_sequence=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries

        self.mask_layer = tf.keras.layers.Masking(mask_value=mask_value)
        self.backbone = backbone or ResNet50Backbone(name='backbone/0/body')
        self.downsample_mask = DownsampleMasking()
        self.transformer = transformer or Transformer(return_intermediate_dec=True,
                                                      name='transformer')
        self.model_dim = self.transformer.model_dim

        self.pos_encoder = pos_encoder or PositionEmbeddingSine(
            num_pos_features=self.model_dim // 2, normalize=True)

        self.input_proj = Conv2D(self.model_dim, kernel_size=1, name='input_proj')

        self.query_embed = tf.Variable(
            tf.zeros((self.num_queries, self.model_dim), dtype=tf.float32),
            name='query_embed/kernel')

        self.class_embed = Linear(num_classes + 1, name='class_embed')

        self.bbox_embed = tf.keras.Sequential([
            Linear(self.model_dim, name='layers/0'),
            ReLU(),
            Linear(self.model_dim, name='layers/1'),
            ReLU(),
            Linear(4, name='layers/2')
        ], name='bbox_embed')

        self.return_decode_sequence = return_decode_sequence

        # Make every layer propagate the mask unchanged by default
        set_supports_masking(self, verbose=False)

    def call(self, inputs, training=False, post_process=False):
        # inputs shape [batch_size, img_h, img_w, c]
        x = self.mask_layer(inputs)  # [batch_size, img_h, img_w, c]
        x = self.backbone(x, training=training)  # [batch_size, h, w, 2048]
        x = self.downsample_mask(x, mask=x._keras_mask)  # [batch_size, h, w, 2048]
        pos_encoding = self.pos_encoder(x)  # [batch_size, h, w, model_dim]

        x = self.input_proj(x)  # [batch_size, h, w, model_dim]
        hs = self.transformer(x, tf.logical_not(x._keras_mask),
                              self.query_embed,
                              pos_encoding, training=training)[0]
        # [n_decoder_layers, batch_size, num_queries, model_dim]

        class_pred = self.class_embed(hs)  # [n_decoder_layers, batch_size, num_queries, num_classes]
        box_pred = tf.sigmoid(self.bbox_embed(hs))  # [n_decoder_layers, batch_size, num_queries, 4]

        # get predictions from last decoder layer
        if self.return_decode_sequence:
            class_pred = tf.transpose(class_pred, [1, 0, 2, 3])  # [batch_size, n_decoder_layers, num_queries, num_classes]
            box_pred = tf.transpose(box_pred, [1, 0, 2, 3])  # [batch_size, n_decoder_layers, num_queries, 4]
        else:
            class_pred = class_pred[-1]  # [batch_size, num_queries, num_classes]
            box_pred = box_pred[-1]  # [batch_size, num_queries, 4]

        return tf.concat([box_pred, class_pred], axis=-1)

    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = (None, None, None, 3)
        super().build(input_shape, **kwargs)

    def post_process(self, y_pred):
        boxes = y_pred[..., :4]
        logits = y_pred[..., 4:]

        probs = tf.nn.softmax(logits, axis=-1)[..., :-1]
        scores = tf.reduce_max(probs, axis=-1)
        labels = tf.argmax(probs, axis=-1)
        boxes = box_cxcywh_to_xyxy(boxes)
        return scores, boxes, labels

    def load_from_pickle(self, pickle_file, verbose=False):
        with open(pickle_file, 'rb') as f:
            detr_weights = pickle.load(f)

        for var in self.variables:
            if verbose:
                print('Loading', var.name)
            var = var.assign(detr_weights[var.name[:-2]])
