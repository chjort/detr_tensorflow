import tensorflow as tf
from tensorflow.keras import layers


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
            shape [batch_size, seq_len]. Masks should have `False` for masking and `True` for no masking.
        :param training: Boolean to indicate whether layer is called for training
        :return:
        """
        q, k, v = inputs  # each with shape [batch_size, seq_len, embedding_dim]

        batch_size = tf.shape(q)[0]
        query = self.query_dense(q)  # (batch_size, seq_len_q, embed_dim)
        key = self.key_dense(k)  # (batch_size, seq_len_kv, embed_dim)
        value = self.value_dense(v)  # (batch_size, seq_len_kv, embed_dim)

        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len_kv, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len_kv, projection_dim)

        attention, weights = self.attention(query, key, value, mask, training)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len_q, embed_dim)
        output = self.projection_dense(concat_attention)  # (batch_size, seq_len_q, embed_dim)

        return output

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value, mask=None, training=None):
        """
        :param query: shape [batch_size, num_heads, seq_len_q, projection_dim]
        :param key: shape [batch_size, num_heads, seq_len_kv, projection_dim]
        :param value: [batch_size, num_heads, seq_len_kv, projection_dim]
        :param mask: list of mask for query, key and value respectively: [query_mask, key_mask, value_mask].
            shapes [batch_size, seq_len_q], [batch_size, seq_len_kv], [batch_size, seq_len_kv]
        :param training:
        """
        q_mask = mask[0] if mask else None
        kv_mask = mask[1] if mask else None

        scores = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = scores / tf.math.sqrt(dim_key)

        scores = self._mask_scores(scores, kv_mask)  # mask the scores
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        output = tf.matmul(weights, value)

        if q_mask is not None:
            q_mask = q_mask[:, tf.newaxis, :, tf.newaxis]  # shape [batch_size, 1, seq_len_q, 1]
            q_mask = tf.cast(q_mask, tf.float32)
            output = output * q_mask

        return output, weights

    def _mask_scores(self, scores, kv_mask):

        if kv_mask is not None:
            kv_mask = kv_mask[:, tf.newaxis, tf.newaxis, :]  # shape [batch_size, 1, 1, seq_len_kv]

        if self.look_ahead_mask:
            la_mask = self._make_look_ahead_mask(tf.shape(scores)[-2:])  # shape [seq_len_q, seq_len_kv]
            la_mask = la_mask[tf.newaxis, tf.newaxis, :, :]  # shape [1, , 1seq_len_q, seq_len_kv]

        else:
            la_mask = None

        scores_mask = self._merge_masks(kv_mask, la_mask)
        if scores_mask is not None:
            scores_mask = tf.logical_not(scores_mask)
            scores_mask = tf.cast(scores_mask, tf.float32)
            scores = scores - 1e-9 * scores_mask

        return scores

    @staticmethod
    def _make_look_ahead_mask(shape):
        mask = tf.linalg.band_part(tf.ones(shape), -1, 0)
        return tf.cast(mask, tf.bool)

    @staticmethod
    def _merge_masks(x, y):
        if x is None:
            return y
        if y is None:
            return x
        return tf.logical_and(x, y)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            q_mask = mask[0]
            if q_mask is None:
                return None
            return tf.convert_to_tensor(q_mask)
        return None

    def get_config(self):
        config = {"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                  "dropout_rate": self.dropout_rate, "look_ahead_mask": self.look_ahead_mask}
        base_config = super(MultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

