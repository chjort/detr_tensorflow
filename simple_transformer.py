import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.1, causal=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causal = causal

        assert embed_dim % num_heads == 0
        depth = embed_dim // num_heads

        self.w_query = tf.keras.layers.Dense(embed_dim)
        self.w_value = tf.keras.layers.Dense(embed_dim)
        self.w_key = tf.keras.layers.Dense(embed_dim)
        self.attention = tf.keras.layers.Attention(causal=causal, dropout=dropout_rate)
        self.projection = tf.keras.layers.Dense(embed_dim)

        self.reshape = tf.keras.layers.Reshape((-1, num_heads, depth))
        self.permute = tf.keras.layers.Permute((2, 1, 3))

        self.reshape_mask = tf.keras.layers.Reshape((-1, 1))
        self.permute_mask = tf.keras.layers.Permute((2, 1))

        self.permute_attention = tf.keras.layers.Permute((2, 1, 3))
        self.reshape_attention = tf.keras.layers.Reshape((-1, embed_dim))

    def call(self, inputs, mask=None, training=None):
        q = inputs[0]  # [batch_size, tq, embed_dim]
        v = inputs[1]  # [batch_size, tv, embed_dim]
        k = inputs[2] if len(inputs) > 2 else v  # [batch_size, tv, embed_dim]

        query = self.w_query(q)
        query = self.reshape(query)  # [batch_size, tq, num_heads, embed_dim]
        query = self.permute(query)  # [batch_size, num_heads, tq, embed_dim]

        value = self.w_value(v)
        value = self.reshape(value)  # [batch_size, tv, num_heads, embed_dim]
        value = self.permute(value)  # [batch_size, num_heads, tv, embed_dim]

        key = self.w_key(k)
        key = self.reshape(key)  # [batch_size, tk, num_heads, embed_dim]
        key = self.permute(key)  # [batch_size, num_heads, tv, embed_dim]

        if mask is not None:
            if mask[0] is not None:
                query_mask = mask[0]  # [batch_size, tq]
                query_mask = self.reshape_mask(query_mask)  # [batch_size, tq, num_heads]
                query_mask = self.permute_mask(query_mask)  # [batch_size, num_heads, tq]
            if mask[1] is not None:
                value_mask = mask[1]  # [batch_size, tv]
                value_mask = self.reshape_mask(value_mask)  # [batch_size, tv, num_heads]
                value_mask = self.permute_mask(value_mask)  # [batch_size, num_heads, tv]

        attention = self.attention([query, value, key], mask=[query_mask, value_mask])
        attention = self.permute_attention(attention)
        attention = self.reshape_attention(attention)

        x = self.projection(attention)

        return x

    def compute_mask(self, inputs, mask=None):
        if mask:
            q_mask = mask[0]
            if q_mask is None:
                return None
            return tf.convert_to_tensor(q_mask)
        return None

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "dropout_rate": self.dropout_rate, "causal": self.causal}
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.dropout_attention = tf.keras.layers.Dropout(dropout_rate)
        self.add_attention = tf.keras.layers.Add()
        self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = tf.keras.layers.Dense(ff_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(embed_dim)
        self.dropout_dense = tf.keras.layers.Dropout(dropout_rate)
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        attention = self.multi_head_attention([inputs, inputs, inputs], mask=[mask, mask])
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

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate}
        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout_rate=0.1, causal=True):
        super(DecoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.causal = causal

        self.multi_head_attention1 = MultiHeadAttention(embed_dim, num_heads, dropout_rate, causal=causal)
        self.dropout_attention1 = tf.keras.layers.Dropout(dropout_rate)
        self.add_attention1 = tf.keras.layers.Add()
        self.layer_norm_attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_attention2 = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.dropout_attention2 = tf.keras.layers.Dropout(dropout_rate)
        self.add_attention2 = tf.keras.layers.Add()
        self.layer_norm_attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = tf.keras.layers.Dense(ff_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(embed_dim)
        self.dropout_dense = tf.keras.layers.Dropout(dropout_rate)
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None, training=None):
        x, x_enc = inputs

        attention = self.multi_head_attention1([x, x, x], mask=[mask[0], mask[0]])
        attention = self.dropout_attention1(attention, training=training)
        x = self.add_attention1([x, attention])
        x = self.layer_norm_attention1(x)

        attention = self.multi_head_attention2([x, x_enc, x_enc], mask=[mask[0], mask[1]])
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

    def compute_mask(self, inputs, mask=None):
        if mask:
            target_mask = mask[0]
            if target_mask is None:
                return None
            return tf.convert_to_tensor(target_mask)
        return None

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate,
                  "causal": self.causal}
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BaseEncoder(tf.keras.layers.Layer):
    def __init__(self, layers, norm=False, **kwargs):
        super(BaseEncoder, self).__init__(**kwargs)
        self.norm = norm
        if norm:
            self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self.norm_layer = None
        self.layers = layers
        self.supports_masking = True

    def call(self, inputs, mask=None, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x, mask=mask)

        if self.norm:
            x = self.norm_layer(x)

        return x

    def get_config(self):
        config = {"norm": self.norm}
        base_config = super(BaseEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BaseDecoder(tf.keras.layers.Layer):
    def __init__(self, layers, norm=False, return_sequence=False, **kwargs):
        super(BaseDecoder, self).__init__(**kwargs)
        self.norm = norm
        if norm:
            self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self.norm_layer = None
        self.return_sequence = return_sequence
        self.layers = layers
        self.supports_masking = True

    def call(self, inputs, mask=None, **kwargs):
        x, enc_output = inputs

        decode_sequence = []
        for layer in self.layers:
            x = layer([x, enc_output], mask=mask)
            decode_sequence.append(x)

        if self.norm:
            decode_sequence = [self.norm_layer(x) for x in decode_sequence]

        if self.return_sequence:
            x = tf.stack(decode_sequence, axis=0)
            x = tf.transpose(x, [1, 0, 2, 3])
        else:
            x = decode_sequence[-1]

        return x

    def compute_mask(self, inputs, mask=None):
        if mask:
            target_mask = mask[0]
            if target_mask is None:
                return None
            return tf.convert_to_tensor(target_mask)
        return None

    def get_config(self):
        config = {"norm": self.norm, "return_sequence": self.return_sequence}
        base_config = super(BaseDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Encoder(BaseEncoder):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1, norm=False, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        layers = [EncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
                  for i in range(num_layers)]
        super(Encoder, self).__init__(layers=layers, norm=norm, **kwargs)

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate,
                  "num_layers": self.num_layers}
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(BaseDecoder):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1, norm=False,
                 causal=True, return_sequence=False, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.causal = causal
        layers = [DecoderLayer(embed_dim, num_heads, ff_dim, dropout_rate, causal)
                  for i in range(num_layers)]
        super(Decoder, self).__init__(layers=layers, norm=norm, return_sequence=return_sequence, **kwargs)

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate,
                  "num_layers": self.num_layers, "causal": self.causal}
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
