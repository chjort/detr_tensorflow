import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from chambers.layers.embedding import PositionalEmbedding1D
from chambers.layers.transformer import TransformerEncoder, TransformerDecoder
from simple_transformer import Encoder, Decoder


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def show_attention_mask(query_mask, key_mask, qk_mask, title=None):
    # qk_mask = tf.matmul(query_mask[..., None], key_mask[..., None], transpose_b=True)
    batch_size = np.minimum(16, query_mask.shape[0])
    fig_ncols = batch_size // 2

    fig = plt.figure()
    gs = fig.add_gridspec(3, fig_ncols)

    ax1 = fig.add_subplot(gs[0, :fig_ncols // 2])
    ax1.imshow(query_mask[:batch_size])

    ax2 = fig.add_subplot(gs[0, fig_ncols // 2:])
    ax2.imshow(key_mask[:batch_size])

    for i in range(0, fig_ncols):
        ax3 = fig.add_subplot(gs[1, i])
        ax3.imshow(qk_mask[i])
        ax3.axis("off")

    for i in range(0, fig_ncols):
        ax4 = fig.add_subplot(gs[2, i])
        ax4.imshow(qk_mask[fig_ncols + i])
        ax4.axis("off")

    plt.subplots_adjust(hspace=0.8)
    plt.suptitle(title)
    plt.show()


def encode(pt, en):
    pt = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        pt.numpy()) + [tokenizer_pt.vocab_size + 1]

    en = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        en.numpy()) + [tokenizer_en.vocab_size + 1]

    return pt, en


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


def make_shifted_target(pt, en):
    input_en = en[:-1]
    target_en = en[1:]

    return (pt, input_en), target_en


# %%
td, info = tfds.load("ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True,
                     data_dir="/datadrive/crr/tensorflow_datasets")

n_train = info.splits["train"].num_examples

train_td = td["train"]
val_td = td["validation"]

# %%
# tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (en.numpy() for pt, en in train_td), target_vocab_size=2 ** 13
# )
# tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (pt.numpy() for pt, en in train_td), target_vocab_size=2 ** 13
# )

# tokenizers will not use 0 as index for subwords. This free up 0 to be used for padding.
tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizer_en")
tokenizer_pt = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizer_pt")

# %%
SHUFFLE_BUFFER = 20000
BATCH_SIZE = 64
MAX_LEN = 40


def filter_max_length(x, y, max_length=MAX_LEN):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


train_td = train_td.map(tf_encode)
train_td = train_td.filter(filter_max_length)
train_td = train_td.map(make_shifted_target)
# cache the dataset to memory to get a speedup while reading from it.
train_td = train_td.cache()
train_td = train_td.repeat()
train_td = train_td.shuffle(SHUFFLE_BUFFER).padded_batch(BATCH_SIZE)
train_td = train_td.prefetch(tf.data.experimental.AUTOTUNE)

val_td = val_td.map(tf_encode)
val_td = val_td.filter(filter_max_length)
val_td = val_td.map(make_shifted_target)
val_td = val_td.padded_batch(BATCH_SIZE)


# %%

# def Seq2SeqTransformer(input_vocab_size, output_vocab_size, embed_dim, num_heads, dim_feedforward,
#                        num_encoder_layers, num_decoder_layers, dropout_rate=0.1, name="detr"):
#     enc_inputs = tf.keras.layers.Input(shape=(None,), name="pt_token_indices")
#     dec_inputs = tf.keras.layers.Input(shape=(None,), name="en_token_indices")
#
#     pos_enc1d = PositionalEmbedding1D(embed_dim)
#
#     x_enc = tf.keras.layers.Embedding(input_vocab_size, embed_dim, mask_zero=True, name="pt_embed")(enc_inputs)
#     x_enc = pos_enc1d(x_enc)
#     x = TransformerEncoder(embed_dim=embed_dim,
#                            num_heads=num_heads,
#                            ff_dim=dim_feedforward,
#                            num_layers=num_encoder_layers,
#                            dropout_rate=dropout_rate)(x_enc)
#     x_dec = tf.keras.layers.Embedding(output_vocab_size, embed_dim, mask_zero=True, name="en_embed")(dec_inputs)
#     x_dec = pos_enc1d(x_dec)
#     x = TransformerDecoder(embed_dim=embed_dim,
#                            num_heads=num_heads,
#                            ff_dim=dim_feedforward,
#                            num_layers=num_decoder_layers,
#                            dropout_rate=dropout_rate,
#                            look_ahead_mask=True)([x_dec, x])
#
#     x = tf.keras.layers.Dense(output_vocab_size)(x)
#
#     model = tf.keras.Model([enc_inputs, dec_inputs], x, name=name)
#
#     return model


# +2 for [START] and [END] token
input_vocab_size = tokenizer_pt.vocab_size + 2
output_vocab_size = tokenizer_en.vocab_size + 2
d_model = 128

inputs = tf.keras.layers.Input(shape=(None,))
targets = tf.keras.layers.Input(shape=(None,))

x_enc = inputs
x_dec = targets

# pos_enc1d = PositionalEmbedding1D(d_model)
#
# x_enc = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True, name="pt_embed")(inputs)
# x_enc = pos_enc1d(x_enc)
#
# x_dec = tf.keras.layers.Embedding(output_vocab_size, d_model, mask_zero=True, name="en_embed")(targets)
# x_dec = pos_enc1d(x_dec)

encoder = Encoder(input_vocab_size, num_layers=4, d_model=d_model, num_heads=8, dff=512, dropout=0.1)
decoder = Decoder(output_vocab_size, num_layers=4, d_model=d_model, num_heads=8, dff=512, dropout=0.1)

x = encoder(x_enc)
x = decoder([x_dec, x], mask=encoder.compute_mask(inputs))
x = tf.keras.layers.Dense(output_vocab_size)(x)

model = tf.keras.models.Model(inputs=[inputs, targets], outputs=x)
model.summary()

# model = Seq2SeqTransformer(input_vocab_size=input_vocab_size,
#                            output_vocab_size=output_vocab_size,
#                            embed_dim=d_model,
#                            num_heads=8,
#                            dim_feedforward=512,
#                            num_encoder_layers=4,
#                            num_decoder_layers=4,
#                            dropout_rate=0.1)
#
# model.summary()

# %%
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics
              )

epochs = 20
steps_per_epoch = 702  # n_train // BATCH_SIZE
model.fit(train_td,
          epochs=epochs,
          steps_per_epoch=steps_per_epoch,
          validation_data=val_td
          )


# %%
class Seq2seqTranslater:
    def __init__(self, input_tokenizer, output_tokenizer, model, maxlen=None):
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.model = model
        self.maxlen = maxlen

        self._input_start_token = [input_tokenizer.vocab_size]
        self._input_end_token = [input_tokenizer.vocab_size + 1]
        self._output_start_token = [output_tokenizer.vocab_size]
        self._output_end_token = [output_tokenizer.vocab_size + 1]

    def translate(self, seq):
        seq_tokens = self._input_start_token + self.input_tokenizer.encode(seq) + self._input_end_token

        # add batch dimension
        encoder_input = tf.expand_dims(seq_tokens, 0)
        decode_sequence = tf.expand_dims(self._output_start_token, 0)

        prediction_token = None
        i = 0
        while prediction_token != self._output_end_token[0]:
            prediction = self.model([encoder_input, decode_sequence], training=False)
            prediction = prediction[:, -1:, :]  # select the last word from the seq_len dimension
            prediction_token = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
            decode_sequence = tf.concat([decode_sequence, prediction_token], axis=-1)
            i = i + 1

            if self.maxlen is not None and i > self.maxlen:
                break

        decode_sequence = decode_sequence[0]  # remove batch dimension

        predicted_sentence = self.output_tokenizer.decode([i for i in decode_sequence if i < self._output_start_token])
        return predicted_sentence


translater = Seq2seqTranslater(tokenizer_pt, tokenizer_en, model, maxlen=MAX_LEN)

input_sequence = "este é um problema que temos que resolver."
output_sequence = translater.translate(input_sequence)
print(output_sequence)
