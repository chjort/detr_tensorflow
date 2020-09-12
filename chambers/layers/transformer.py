import tensorflow as tf

from .attention import MultiHeadSelfAttention


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate, name="self_attn")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")

        self.linear1 = tf.keras.layers.Dense(ff_dim, activation="relu", name="linear1")
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.linear2 = tf.keras.layers.Dense(embed_dim, name="linear2")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.supports_masking = True

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


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
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
        self.supports_masking = True

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


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dim_feedforward, num_layers, dropout_rate=0.1, norm=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.norm = norm
        if norm:
            self._norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self._norm_layer = None
        self.layers = [TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout_rate)
                       for i in range(num_layers)]
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        if self.norm:
            x = self._norm_layer(x)

        return x


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dim_feedforward, num_layers, dropout_rate=0.1, norm=False,
                 return_sequence=False, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.norm = norm
        if norm:
            self._norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self._norm_layer = None
        self.return_sequence = return_sequence
        self.layers = [TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout_rate)
                       for i in range(num_layers)]

    def call(self, inputs, **kwargs):
        x, x_enc = inputs

        decode_sequence = []
        for layer in self.layers:
            x = layer([x, x_enc])
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