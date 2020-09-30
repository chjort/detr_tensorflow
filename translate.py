import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from chambers.layers.embedding import PositionalEmbedding1D
from chambers.layers.transformer import TransformerEncoder, TransformerDecoder

# %%
td, info = tfds.load("ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True,
                     data_dir="/datadrive/crr/tensorflow_datasets")

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


# %%
SHUFFLE_BUFFER = 20000
BATCH_SIZE = 16
MAX_LEN = 40


def filter_max_length(x, y, max_length=MAX_LEN):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


train_td = train_td.map(tf_encode)
train_td = train_td.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_td = train_td.cache()
train_td = train_td.shuffle(SHUFFLE_BUFFER).padded_batch(BATCH_SIZE)
train_td = train_td.prefetch(tf.data.experimental.AUTOTUNE)

val_td = val_td.map(tf_encode)
val_td = val_td.filter(filter_max_length).padded_batch(BATCH_SIZE)


# %%
def create_padding_mask_tut(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask_tut(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    seq = 1 - tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = tf.linalg.band_part(tf.ones(size), -1, 0)
    return mask  # (seq_len, seq_len)


# %%
input_vocab_size = 8500
embed_dim = 512

enc_inputs = tf.keras.layers.Input(shape=(None,), name="en_token_indices")
dec_inputs = tf.keras.layers.Input(shape=(None,), name="pt_token_indices")

pos_enc1d = PositionalEmbedding1D(embed_dim)

x_enc = tf.keras.layers.Embedding(input_vocab_size, embed_dim, mask_zero=True)(enc_inputs)
x_enc = pos_enc1d(x_enc)
x = TransformerEncoder(embed_dim=embed_dim,
                       num_heads=8,
                       ff_dim=2048,
                       num_layers=2,
                       dropout_rate=0.1)(x_enc)
x_dec = tf.keras.layers.Embedding(input_vocab_size, embed_dim, mask_zero=True)(dec_inputs)
x_dec = pos_enc1d(x_dec)
x = TransformerDecoder(embed_dim=embed_dim,
                       num_heads=8,
                       ff_dim=2048,
                       num_layers=2,
                       dropout_rate=0.1)([x_dec, x])

# model = tf.keras.Model([enc_inputs, dec_inputs], x)
model = tf.keras.Model([enc_inputs, dec_inputs], [x_enc, x_dec])
model.summary()

# %%
it = iter(val_td)
x1, x2 = next(it)
print(x1.shape, x2.shape)

z1, z2 = model([x1, x2])

m1 = tf.cast(z1._keras_mask, tf.float32)
m2 = tf.cast(z2._keras_mask, tf.float32)
print(m1.shape, m2.shape)

# %%
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


# %% Self attention

sattw = tf.matmul(z2, z2, transpose_b=True)
sattw.shape

msa = tf.matmul(m2[..., None], m2[..., None],
                transpose_b=True)  # TODO: How do we get attention mask from query mask and key mask?
msa.shape

m_sattw = sattw + (-1e9 * (1.0 - msa))

# show_attention_mask(m2, m2, msa)
show_attention_mask(m2, m2, m_sattw, "Self-attention (ch)")

# %%
msa_o = tf.matmul(m2[..., None], tf.ones_like(m2)[..., None],
                transpose_b=True)  # TODO: How do we get attention mask from query mask and key mask?
msa_o.shape

m_sattw_o = sattw + (-1e9 * (1.0 - msa_o))
show_attention_mask(m2, m2, m_sattw_o, "Self-attention: key_mask=Ones (ch)")

# %%
msa_o = tf.matmul(tf.ones_like(m2)[..., None], m2[..., None],
                  transpose_b=True)  # TODO: How do we get attention mask from query mask and key mask?
msa_o.shape

m_sattw_o = sattw + (-1e9 * (1.0 - msa_o))
show_attention_mask(m2, m2, m_sattw_o, "Self-attention: query_mask=Ones (ch)")

# %% Self attention (look ahead mask)
sattw_l = tf.matmul(z2, z2, transpose_b=True)
sattw_l.shape

msa_l = tf.matmul(m2[..., None], m2[..., None],
                  transpose_b=True)  # TODO: How do we get attention mask from query mask and key mask?
msa_l.shape

lam = create_look_ahead_mask(sattw_l.shape[-2:])
msa_l = msa_l * lam

m_sattw_l = sattw_l + (-1e9 * (1.0 - msa_l))

# show_attention_mask(m2, m2, msa_l)
show_attention_mask(m2, m2, m_sattw_l, "Self-attention: look ahead mask (ch)")

# %% attention
attw = tf.matmul(z1, z2, transpose_b=True)
attw.shape

ma = tf.matmul(m1[..., None], m2[..., None],
               transpose_b=True)  # TODO: How do we get attention mask from query mask and key mask?
ma.shape

m_attw = attw + (-1e9 * (1.0 - ma))

# show_attention_mask(m1, m2, ma)
show_attention_mask(m1, m2, m_attw, "Attention (ch)")

# %% self-attention tutorial (look ahead mask)
enc_pad_mask = create_padding_mask(x1)
dec_target_pad_mask = create_padding_mask(x2)
enc_pad_mask.shape
dec_target_pad_mask.shape
lam.shape

combined_mask = tf.minimum(dec_target_pad_mask, lam)
combined_mask.shape

show_attention_mask(m2, m2, combined_mask[:, 0, :, :], "Self-attention: look ahead mask (tut)")

# %% self-attention tutorial
sattw_h = sattw[:, None, :, :]
sattw_hm = sattw_h + (-1e9 * (1.0 - dec_target_pad_mask))
show_attention_mask(m2, m2, sattw_hm[:, 0, :, :], "Self-attention: key_mask=None (tut)")

# %%
# a = tf.constant([[1, 1, 0, 0], [1, 0, 0, 0]])
# b = tf.constant([[0, 1, 0, 0], [1, 1, 1, 0]])
#
# a = tf.expand_dims(a, -1)
# b = tf.expand_dims(b, -1)
#
#
# print(a.shape, b.shape)
#
# print(tf.matmul(a, b, transpose_b=True))

# %%
# print(z.shape)
# print(z._keras_mask.shape)

# tm = tf.cast(z._keras_mask, tf.float32)

# lam = create_look_ahead_mask(x2.shape[1])
# m = create_padding_mask(x2)

# tm  # decoder input padding mask (from decoder input embedding layer)
# lam  # look-ahead mask
# m

# lam = tf.ones_like(lam)

# lam.shape, m.shape, tm.shape
# combined_mask = tf.minimum(lam, m)[:, 0, :, :]  # Attention weights mask
# combined_mask.shape
# combined_mask
# combined_mask[0]
# tf.assert_equal(combined_mask, cm)
"""
array([[1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.],
       ...,
       [1., 1., 1., ..., 1., 0., 0.],
       [1., 1., 1., ..., 1., 1., 0.],
       [1., 1., 1., ..., 1., 1., 1.]], dtype=float32)>
       
TARGET
array([[1., 0., 0., ..., 0., 0., 0.],
       [1., 1., 0., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.],
       ...,
       [1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.]], dtype=float32)>
"""
