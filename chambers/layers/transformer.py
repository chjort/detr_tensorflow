import tensorflow as tf

from .attention import MultiHeadSelfAttention


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, name="self_attn")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")

        self.linear1 = tf.keras.layers.Dense(ff_dim, activation="relu", name="linear1")
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.linear2 = tf.keras.layers.Dense(embed_dim, name="linear2")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")

    def call(self, inputs, training=None):
        # inputs.shape = [batch_size, sequence_length, embed_dim]

        # TODO: MASKING
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
        self.attn1 = MultiHeadSelfAttention(embed_dim, num_heads, name="self_attn")
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")

        self.attn2 = MultiHeadSelfAttention(embed_dim, num_heads, name="multihead_attn")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")

        self.linear1 = tf.keras.layers.Dense(ff_dim, activation="relu", name="linear1")
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        self.linear2 = tf.keras.layers.Dense(embed_dim, name="linear2")
        self.dropout4 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm3")

    def call(self, inputs, training=None):
        x, enc_output = inputs

        # TODO: MASKING

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


# class TransformerEncoder(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, dim_feedforward, num_layers, dropout_rate=0.1, **kwargs):
#         super(TransformerEncoder, self).__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dim_feedforward = dim_feedforward
#         self.num_layers = num_layers
#         self.layers = [TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout_rate)
#                        for i in range(num_layers)]


class TransformerDecoderDETR(tf.keras.layers.Layer):
    def __init__(self, output_len, embed_dim, num_heads, dim_feedforward, num_layers, dropout_rate=0.1,
                 return_sequence=False, **kwargs):
        super(TransformerDecoderDETR, self).__init__(**kwargs)
        self.output_len = output_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.return_sequence = return_sequence
        self.layers = [TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout_rate)
                       for i in range(num_layers)]

        # self.decoder_embedding = self.add_weight(name="decoder_embeddings",
        #                                          shape=(output_len, embed_dim),
        #                                          dtype=tf.float32,
        #                                          initializer="zeros",
        #                                          trainable=True)
        self.decoder_emb = tf.keras.layers.Embedding(output_len, 256, embeddings_initializer="zeros")

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        x = tf.tile(tf.expand_dims(tf.range(self.output_len), 0), [batch_size, 1])
        x = self.decoder_emb(x)

        decode_sequence = []
        for layer in self.layers:
            x = layer([x, inputs])
            decode_sequence.append(x)

        if self.return_sequence:
            x = tf.stack(decode_sequence, axis=0)

        return x
