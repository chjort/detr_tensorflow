import tensorflow as tf

from chambers.layers.transformer import TransformerEncoderLayer, TransformerDecoderDETR, TransformerDecoderLayer

"""
def __init__(self,
             model_dim: int = 256,
             num_heads: int = 8,
             num_encoder_layers: int = 6,
             num_decoder_layers: int = 6,
             dim_feedforward: int = 2048,
             dropout: float = 0.1,
             activation: str = 'relu',
             normalize_before: bool = False,
             return_intermediate_dec: bool = False,
             **kwargs: Any) -> None

"""


def TransformerDETR(input_shape, output_len, num_heads, dim_feedforward,
                    num_encoder_layers, num_decoder_layers,
                    dropout_rate=0.1, return_decode_sequence=False):
    input_ = tf.keras.layers.Input(shape=input_shape)
    embed_dim = input_shape[1]

    x_enc = input_
    for i in range(num_encoder_layers):
        x_enc = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout_rate)(x_enc)

    batch_size = tf.shape(x_enc)[0]
    x = tf.tile(tf.expand_dims(tf.range(output_len), 0), [batch_size, 1])
    x = tf.keras.layers.Embedding(output_len, embed_dim, embeddings_initializer="zeros")(x)
    decode_sequence = []
    for i in range(num_decoder_layers):
        x = TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout_rate)([x, x_enc])
        decode_sequence.append(x)

    if return_decode_sequence:
        x = tf.stack(decode_sequence, axis=0)

    # x = TransformerDecoderDETR(output_len, embed_dim, num_heads, dim_feedforward, num_decoder_layers,
    #                            dropout_rate, return_decode_sequence)(x_enc)

    model = tf.keras.models.Model(input_, x)

    return model


#%%
input_shape = (1064, 256)
output_len = 100
num_heads = 8
dim_feedforward = 2048
num_encoder_layers = 6
num_decoder_layers = 6
dropout_rate = 0.1
return_decode_sequence = False

x = tf.random.normal(shape=(2, 28, 38, 256))
x = tf.reshape(x, [2, -1, 256])

# %%
model = TransformerDETR(input_shape=(1064, 256),
                        output_len=100,
                        num_heads=8,
                        dim_feedforward=2048,
                        num_encoder_layers=6,
                        num_decoder_layers=6,
                        dropout_rate=0.1,
                        return_decode_sequence=False
                        )

model.summary()

z = model(x)
print(z.shape)