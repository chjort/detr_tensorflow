import pickle

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU

from chambers.utils.tf import set_supports_masking
from utils import cxcywh2xyxy
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
                 **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries

        self.mask_layer = tf.keras.layers.Masking(mask_value=mask_value)
        self.backbone = backbone or ResNet50Backbone(name='backbone/0/body')
        self.transformer = transformer or Transformer(return_intermediate_dec=True,
                                                      name='transformer')
        self.model_dim = self.transformer.model_dim

        self.pos_encoder = pos_encoder or PositionEmbeddingSine(
            num_pos_features=self.model_dim // 2, normalize=True)

        self.input_proj = Conv2D(self.model_dim, kernel_size=1, name='input_proj')

        self.query_embed = tf.Variable(
            tf.zeros((num_queries, self.model_dim), dtype=tf.float32),
            name='query_embed/kernel')

        self.class_embed = Linear(num_classes + 1, name='class_embed')

        self.bbox_embed = tf.keras.Sequential([
            Linear(self.model_dim, name='layers/0'),
            ReLU(),
            Linear(self.model_dim, name='layers/1'),
            ReLU(),
            Linear(4, name='layers/2')
        ], name='bbox_embed')
        set_supports_masking(self)

    def call(self, inputs, training=False, post_process=False):
        x = self.mask_layer(inputs)
        x = self.backbone(x, training=training)
        pos_encoding = self.pos_encoder(x)

        x = self.input_proj(x)
        hs = self.transformer(x, tf.logical_not(x._keras_mask),
                              self.query_embed,
                              pos_encoding, training=training)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = tf.sigmoid(self.bbox_embed(hs))

        output = {'pred_logits': outputs_class[-1],
                  'pred_boxes': outputs_coord[-1]}

        if post_process:
            output = self.post_process(output)
        return output

    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = (None, None, None, 3)
        super().build(input_shape, **kwargs)

    def post_process(self, output):
        logits, boxes = [output[k] for k in ['pred_logits', 'pred_boxes']]

        probs = tf.nn.softmax(logits, axis=-1)[..., :-1]
        scores = tf.reduce_max(probs, axis=-1)
        labels = tf.argmax(probs, axis=-1)
        boxes = cxcywh2xyxy(boxes)

        output = {'scores': scores,
                  'labels': labels,
                  'boxes': boxes}
        return output

    def load_from_pickle(self, pickle_file, verbose=False):
        with open(pickle_file, 'rb') as f:
            detr_weights = pickle.load(f)

        for var in self.variables:
            if verbose:
                print('Loading', var.name)
            var = var.assign(detr_weights[var.name[:-2]])
