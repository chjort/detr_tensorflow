import tensorflow as tf
from tensorflow.keras import layers
tf.keras.layers.Attention

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.0, look_ahead_mask=False, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.look_ahead_mask = look_ahead_mask
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
        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        """

        :param inputs: List of [queries, keys, values], where each element has
            shape [batch_size, seq_len, embedding_dim]
        :param mask: List of [queries_mask, keys_mask, values_mask], where each element has
            shape [batch_size, seq_len]. Masks should have `0` for masking and `1` for no masking.
        :param training: Boolean to indicate whether layer is called for training
        :return:
        """
        q, k, v = inputs  # each with shape [batch_size, seq_len, embedding_dim]

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

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value, mask=None, training=None):
        logits = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_logits = logits / tf.math.sqrt(dim_key)

        if mask is not None and (mask[0] is not None or mask[1] is not None):
            query_mask, key_mask = mask[0], mask[1]
            logits_mask = self._compute_weights_mask(query, key, query_mask, key_mask)
            scaled_logits = scaled_logits + (-1e9 * (1.0 - logits_mask))  # apply mask
            tf.print(tf.shape(scaled_logits), tf.shape(logits_mask))

        weights = tf.nn.softmax(scaled_logits, axis=-1)
        weights = self.dropout(weights, training=training)
        output = tf.matmul(weights, value)
        return output, weights

    def _compute_weights_mask(self, query, key, query_mask, key_mask):
        query_seq_len = tf.shape(query)[-2]
        key_seq_len = tf.shape(key)[-2]
        if query_mask is None:
            query_mask = tf.ones([tf.shape(query)[0], query_seq_len], dtype=tf.float32)
        if key_mask is None:
            key_mask = tf.ones([tf.shape(key)[0], key_seq_len], dtype=tf.float32)

        query_mask = tf.expand_dims(tf.cast(query_mask, tf.float32), -1)
        key_mask = tf.expand_dims(tf.cast(key_mask, tf.float32), -1)

        logits_mask = tf.matmul(query_mask, key_mask, transpose_b=True)  # [batch_size, seq_len_q, seq_len_k]

        if self.look_ahead_mask:
            look_ahead_mask = self._make_look_ahead_mask([query_seq_len, key_seq_len])
            logits_mask = logits_mask * look_ahead_mask

        if tf.rank(query) == 4:
            logits_mask = tf.expand_dims(logits_mask, 1)  # [batch_size, 1, seq_len_q, seq_len_k]

        return logits_mask

    def _make_look_ahead_mask(self, shape):
        mask = tf.linalg.band_part(tf.ones(shape), -1, 0)
        return mask

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "dropout_rate": self.dropout_rate, "look_ahead_mask": self.look_ahead_mask}
        base_config = super(MultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
