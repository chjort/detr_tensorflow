import tensorflow as tf
from tensorflow.keras.layers import Attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.1, causal=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causal = causal

        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads

        self.w_query = tf.keras.layers.Dense(embed_dim)
        self.w_value = tf.keras.layers.Dense(embed_dim)
        self.w_key = tf.keras.layers.Dense(embed_dim)
        self.attention = Attention(causal=causal, dropout=dropout_rate)
        self.projection = tf.keras.layers.Dense(embed_dim)

        self.reshape = tf.keras.layers.Reshape((-1, num_heads, head_dim))
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
        query = self.reshape(query)  # [batch_size, tq, num_heads, head_dim]
        query = self.permute(query)  # [batch_size, num_heads, tq, head_dim]

        value = self.w_value(v)
        value = self.reshape(value)  # [batch_size, tv, num_heads, head_dim]
        value = self.permute(value)  # [batch_size, num_heads, tv, head_dim]

        key = self.w_key(k)
        key = self.reshape(key)  # [batch_size, tk, num_heads, head_dim]
        key = self.permute(key)  # [batch_size, num_heads, tv, head_dim]

        if mask is not None:
            if mask[0] is not None:
                query_mask = mask[0]  # [batch_size, tq]
                query_mask = self.reshape_mask(query_mask)  # [batch_size, tq, num_heads]
                query_mask = self.permute_mask(query_mask)  # [batch_size, num_heads, tq]
            else:
                query_mask = None

            if mask[1] is not None:
                value_mask = mask[1]  # [batch_size, tv]
                value_mask = self.reshape_mask(value_mask)  # [batch_size, tv, num_heads]
                value_mask = self.permute_mask(value_mask)  # [batch_size, num_heads, tv]
            else:
                value_mask = None

        else:
            query_mask = None
            value_mask = None

        attention = self.attention([query, value, key], mask=[query_mask, value_mask])  # [batch_size, num_heads, tq, head_dim]
        attention = self.permute_attention(attention)  # [batch_size, tq, num_heads, head_dim]
        attention = self.reshape_attention(attention)  # [batch_size, tq, embed_dim]
        x = self.projection(attention)  # [batch_size, tq, embed_dim]

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
