import tensorflow as tf
from tensorflow.keras import layers


class Attention(layers.Layer):
    def __init__(self, embed_dim, dropout_rate=0.0, return_weights=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.return_weights = return_weights
        self.query_linear = tf.keras.layers.Dense(embed_dim)
        self.key_linear = tf.keras.layers.Dense(embed_dim)
        self.value_linear = tf.keras.layers.Dense(embed_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, mask=None, training=None):
        # mask shape (batch_size, seq_len_q, seq_len_k). `0` for masking and `1` for no masking

        query, key, value = inputs
        # q, k, v shape = [batch_size, seq_len, embedding_dim]

        query = self.query_linear(query)
        key = self.query_linear(key)
        value = self.query_linear(value)

        logits = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_logits = logits / tf.math.sqrt(dim_key)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            scaled_logits = scaled_logits + (-1e9 * (1.0 - mask))

        weights = tf.nn.softmax(scaled_logits, axis=-1)
        weights = self.dropout(weights, training=training)
        output = tf.matmul(weights, value)

        if self.return_weights:
            return output, weights
        else:
            return output

    def compute_mask(self, inputs, mask=None):
        return mask


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.0, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.projection_dense = layers.Dense(embed_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def attention(self, query, key, value, mask=None, training=None):
        logits = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_logits = logits / tf.math.sqrt(dim_key)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            scaled_logits = scaled_logits + (-1e9 * (1.0 - mask))

        weights = tf.nn.softmax(scaled_logits, axis=-1)
        weights = self.dropout(weights, training=training)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, training=None):
        # mask shape (batch_size, num_heads, seq_len_q, seq_len_k). `0` for masking and `1` for no masking

        q, k, v = inputs
        # q, k, v shape = [batch_size, seq_len, embedding_dim]

        batch_size = tf.shape(q)[0]
        query = self.query_dense(q)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(k)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(v)  # (batch_size, seq_len, embed_dim)

        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)

        attention, weights = self.attention(query, key, value, mask, training)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.projection_dense(concat_attention)  # (batch_size, seq_len, embed_dim)

        return output

    def compute_mask(self, inputs, mask=None):
        return mask
