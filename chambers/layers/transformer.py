import tensorflow as tf

from .attention import MultiHeadSelfAttention


class BaseTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, layers, norm=False, **kwargs):
        super(BaseTransformerEncoder, self).__init__(**kwargs)
        self.norm = norm
        if norm:
            self._norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self._norm_layer = None
        self.layers = layers
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        if self.norm:
            x = self._norm_layer(x)

        return x

    def get_config(self):
        config = {"norm": self.norm}
        base_config = super(BaseTransformerEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BaseTransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, layers, norm=False, return_sequence=False, **kwargs):
        super(BaseTransformerDecoder, self).__init__(**kwargs)
        self.norm = norm
        if norm:
            self._norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self._norm_layer = None
        self.return_sequence = return_sequence
        self.layers = layers

    def call(self, inputs, **kwargs):
        x, enc_output = inputs

        decode_sequence = []
        for layer in self.layers:
            x = layer([x, enc_output])
            decode_sequence.append(x)

        if self.norm:
            x = self._norm_layer(x)
            decode_sequence = [self._norm_layer(x) for x in decode_sequence]

        if self.return_sequence:
            x = tf.stack(decode_sequence, axis=0)
            x = tf.transpose(x, [1, 0, 2, 3])

        return x

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {"norm": self.norm, "return_sequence": self.return_sequence}
        base_config = super(BaseTransformerDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.supports_masking = True

        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate, name="self_attn")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")

        self.linear1 = tf.keras.layers.Dense(ff_dim, activation="relu", name="linear1")
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.linear2 = tf.keras.layers.Dense(embed_dim, name="linear2")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")

    def call(self, inputs, training=None):
        # inputs.shape = [batch_size, sequence_length, embed_dim]
        attn_output = self.att([inputs, inputs, inputs])
        attn_output = self.dropout(attn_output, training=training)
        norm_output1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.linear1(norm_output1)
        ffn_output = self.dropout1(ffn_output, training=training)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        norm_output2 = self.layernorm2(norm_output1 + ffn_output)

        return norm_output2

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate}
        base_config = super(TransformerEncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.supports_masking = True

        self.attn1 = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate, name="self_attn")
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")

        self.attn2 = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate, name="multihead_attn")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")

        self.linear1 = tf.keras.layers.Dense(ff_dim, activation="relu", name="linear1")
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        self.linear2 = tf.keras.layers.Dense(embed_dim, name="linear2")
        self.dropout4 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm3")

    def call(self, inputs, training=None):
        x, enc_output = inputs

        attn_output1 = self.attn1([x, x, x])
        attn_output1 = self.dropout1(attn_output1, training=training)
        norm_output1 = self.layernorm1(x + attn_output1)

        attn_output2 = self.attn2([norm_output1, enc_output, enc_output])
        attn_output2 = self.dropout2(attn_output2)
        norm_output2 = self.layernorm2(norm_output1 + attn_output2)

        ffn_output = self.linear1(norm_output1)
        ffn_output = self.dropout3(ffn_output, training=training)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout4(ffn_output, training=training)
        norm_output3 = self.layernorm3(norm_output2 + ffn_output)

        return norm_output3

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate}
        base_config = super(TransformerDecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TransformerEncoder(BaseTransformerEncoder):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1, norm=False, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        layers = [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
                  for i in range(num_layers)]
        super(TransformerEncoder, self).__init__(layers=layers, norm=norm, **kwargs)

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate,
                  "num_layers": self.num_layers}
        base_config = super(TransformerEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TransformerDecoder(BaseTransformerDecoder):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1, norm=False,
                 return_sequence=False, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        layers = [TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
                  for i in range(num_layers)]
        super(TransformerDecoder, self).__init__(layers=layers, norm=norm, return_sequence=return_sequence, **kwargs)

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate,
                  "num_layers": self.num_layers}
        base_config = super(TransformerDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
