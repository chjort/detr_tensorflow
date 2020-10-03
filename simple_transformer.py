import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, causal=False, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0
        depth = embed_dim // num_heads

        self.w_query = tf.keras.layers.Dense(embed_dim)
        self.w_value = tf.keras.layers.Dense(embed_dim)
        self.w_key = tf.keras.layers.Dense(embed_dim)
        self.attention = tf.keras.layers.Attention(causal=causal, dropout=dropout)
        self.projection = tf.keras.layers.Dense(embed_dim)

        self.split_reshape = tf.keras.layers.Reshape((-1, num_heads, depth))
        self.split_permute = tf.keras.layers.Permute((2, 1, 3))

        self.split_reshape_mask = tf.keras.layers.Reshape((-1, 1))
        self.split_permute_mask = tf.keras.layers.Permute((2, 1))

        self.join_permute_attention = tf.keras.layers.Permute((2, 1, 3))
        self.join_reshape_attention = tf.keras.layers.Reshape((-1, embed_dim))

    def call(self, inputs, mask=None, training=None):
        q = inputs[0]  # [batch_size, tq, embed_dim]
        v = inputs[1]  # [batch_size, tv, embed_dim]
        k = inputs[2] if len(inputs) > 2 else v  # [batch_size, tk, embed_dim]

        query = self.w_query(q)
        query = self.split_reshape(query)  # [batch_size, tq, num_heads, embed_dim]
        query = self.split_permute(query)  # [batch_size, num_heads, tq, embed_dim]

        value = self.w_value(v)
        value = self.split_reshape(value)  # [batch_size, tv, num_heads, embed_dim]
        value = self.split_permute(value)  # [batch_size, num_heads, tv, embed_dim]

        key = self.w_key(k)
        key = self.split_reshape(key)  # [batch_size, tk, num_heads, embed_dim]
        key = self.split_permute(key)  # [batch_size, num_heads, tk, embed_dim]

        if mask is not None:
            if mask[0] is not None:
                query_mask = mask[0]  # [batch_size, tq]
                query_mask = self.split_reshape_mask(query_mask)  # [batch_size, tq, num_heads]
                query_mask = self.split_permute_mask(query_mask)  # [batch_size, num_heads, tq]
            if mask[1] is not None:
                value_mask = mask[1]  # [batch_size, tv]
                value_mask = self.split_reshape_mask(value_mask)  # [batch_size, tv, num_heads]
                value_mask = self.split_permute_mask(value_mask)  # [batch_size, num_heads, tv]

        attention = self.attention([query, value, key], mask=[query_mask, value_mask])
        attention = self.join_permute_attention(attention)
        attention = self.join_reshape_attention(attention)

        x = self.projection(attention)

        return x

    def compute_mask(self, inputs, mask=None):
        if mask:
            q_mask = mask[0]
            if q_mask is None:
                return None
            return tf.convert_to_tensor(q_mask)
        return None


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, dff=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.dropout_attention = tf.keras.layers.Dropout(dropout)
        self.add_attention = tf.keras.layers.Add()
        self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(embed_dim)
        self.dropout_dense = tf.keras.layers.Dropout(dropout)
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        attention = self.multi_head_attention([inputs, inputs, inputs], mask=[mask, mask])
        attention = self.dropout_attention(attention, training=training)
        x = self.add_attention([inputs, attention])
        x = self.layer_norm_attention(x)

        ## Feed Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training=training)
        x = self.add_dense([x, dense])
        x = self.layer_norm_dense(x)

        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, dff=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(embed_dim, num_heads, causal=True)
        self.dropout_attention1 = tf.keras.layers.Dropout(dropout)
        self.add_attention1 = tf.keras.layers.Add()
        self.layer_norm_attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_attention2 = MultiHeadAttention(embed_dim, num_heads)
        self.dropout_attention2 = tf.keras.layers.Dropout(dropout)
        self.add_attention2 = tf.keras.layers.Add()
        self.layer_norm_attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(embed_dim)
        self.dropout_dense = tf.keras.layers.Dropout(dropout)
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
        x = self.layer_norm_attention1(x)

        ## Feed Forward
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


class Encoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_layers=4, num_heads=8, dff=2048, dropout=0.0):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.encoder_layers = [EncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dff=dff, dropout=dropout) for _ in
                               range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        x = self.dropout(inputs, training=training)

        # Encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask=mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_layers=4, num_heads=8, dff=2048, dropout=0.0):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.decoder_layers = [DecoderLayer(embed_dim=embed_dim, num_heads=num_heads, dff=dff, dropout=dropout) for _ in
                               range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        x, x_enc = inputs
        x = self.dropout(x, training=training)

        # Decoder layer
        for decoder_layer in self.decoder_layers:
            x = decoder_layer([x, x_enc], mask=mask)

        return x

    def compute_mask(self, inputs, mask=None):
        if mask:
            target_mask = mask[0]
            if target_mask is None:
                return None
            return tf.convert_to_tensor(target_mask)
        return None
