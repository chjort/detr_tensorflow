import tensorflow as tf

from chambers.layers.transformer import TransformerEncoder, TransformerDecoder


def Transformer(input_shape, num_heads, dim_feedforward,
                num_encoder_layers, num_decoder_layers,
                dropout_rate=0.1, return_decode_sequence=False, name="transformer"):
    x_enc_shape, x_dec_shape = input_shape
    x_enc = tf.keras.layers.Input(shape=x_enc_shape, name="encoder_input")
    x_dec = tf.keras.layers.Input(shape=x_dec_shape, name="decoder_input")

    embed_dim = x_enc_shape[1]

    x = TransformerEncoder(embed_dim, num_heads, dim_feedforward, num_encoder_layers, dropout_rate, norm=False)(x_enc)
    x = TransformerDecoder(embed_dim, num_heads, dim_feedforward, num_decoder_layers, dropout_rate, norm=True,
                           return_sequence=return_decode_sequence)([x_dec, x])

    model = tf.keras.models.Model([x_enc, x_dec], x, name=name)

    return model
