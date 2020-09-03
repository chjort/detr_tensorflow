import tensorflow as tf
import tensorflow_addons as tfa

from chambers.callbacks import HungarianLossLogger
from chambers.losses import HungarianLoss, pairwise_softmax, pairwise_l1, pairwise_giou
from chambers.optimizers import LearningRateMultiplier
from models import build_detr_resnet50
from tf_datasets import load_coco_tf, load_coco


def loss_placeholder(y_true, y_pred):
    y_true_labels = y_true[..., -1]  # [1]
    y_pred_logits = y_pred[..., 4:]  # [1]
    y_pred = y_pred_logits[:, :1, :]
    y_true = tf.one_hot(tf.cast(y_true_labels, tf.int32), depth=92)[:, :1, :]
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)

# %% strategy
strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
# strategy = tf.distribute.Strategy()

# %%
BATCH_SIZE_PER_REPLICA = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
# GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA

# dataset, N = load_coco_tf("train", GLOBAL_BATCH_SIZE)
dataset, N = load_coco("/datadrive/crr/datasets/coco", "train", GLOBAL_BATCH_SIZE)
# dataset, N = load_coco("/home/ch/datasets/coco", "train", BATCH_SIZE)

# dataset = dataset.prefetch(-1)

# %%
with strategy.scope():
# with tf.device("/gpu:0"):
    decode_sequence = False
    detr = build_detr_resnet50(num_classes=91,
                               num_queries=100,
                               mask_value=-1.,
                               return_decode_sequence=decode_sequence)
    detr.build()
    # detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

    # 150 epoch schedule: lr = lr * 0.1 after 100 epochs
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
                                                                 decay_steps=100,
                                                                 decay_rate=0.1,
                                                                 staircase=True)

    opt = tfa.optimizers.AdamW(weight_decay=1e-4,
                               learning_rate=lr_schedule,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-8,
                               # clipnorm=0.1,
                               amsgrad=False,
                               )

    opt = LearningRateMultiplier(opt, lr_multipliers={"backbone": 0.1})

hungarian = HungarianLoss(lsa_losses=[pairwise_softmax, pairwise_l1, pairwise_giou],
                          lsa_loss_weights=[1, 5, 2],
                          mask_value=-1.,
                          sequence_input=decode_sequence)
detr.compile(optimizer=opt,
             loss=hungarian,
             # loss=loss_placeholder
             )

# %% TRAIN
EPOCHS = 10  # 150
N = 200
STEPS_PER_EPOCH = N / GLOBAL_BATCH_SIZE

# print("Number of devices in strategy:", strategy.num_replicas_in_sync)
print("Global batch size: {}. Per device batch size: {}".format(GLOBAL_BATCH_SIZE, BATCH_SIZE_PER_REPLICA))

# ssh -L 6006:127.0.0.1:6006 crr@40.68.160.55
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir="tb_logs", write_graph=False, update_freq="epoch", profile_batch=0)
history = detr.fit(dataset,
                   epochs=EPOCHS,
                   steps_per_epoch=STEPS_PER_EPOCH,
                   callbacks=[
                       HungarianLossLogger(),
                       # tensorboard
                   ]
                   )

# Single GPU:
"""
10 epochs (placeholder)
50/50 [==============================] - 129s 3s/step - loss: 4.5206
Epoch 2/10
50/50 [==============================] - 101s 2s/step - loss: 4.5180
Epoch 3/10
50/50 [==============================] - 87s 2s/step - loss: 4.5165
Epoch 4/10
50/50 [==============================] - 101s 2s/step - loss: 4.5168
Epoch 5/10
50/50 [==============================] - 95s 2s/step - loss: 4.5160
Epoch 6/10
50/50 [==============================] - 90s 2s/step - loss: 4.5163
Epoch 7/10
50/50 [==============================] - 91s 2s/step - loss: 4.5164
Epoch 8/10
50/50 [==============================] - 100s 2s/step - loss: 4.5164
Epoch 9/10
50/50 [==============================] - 90s 2s/step - loss: 4.5171
Epoch 10/10
50/50 [==============================] - 84s 2s/step - loss: 4.5166

TODO: 10 epochs (placeholder, no prefetch)

10 epochs (hungarian)
50/50 [==============================] - 130s 3s/step - loss: 11.8833 - loss_ce: 4.5189 - loss_l1: 5.2441 - loss_giou: 2.1202
Epoch 2/10
50/50 [==============================] - 101s 2s/step - loss: 11.6155 - loss_ce: 4.5129 - loss_l1: 5.0268 - loss_giou: 2.0758
Epoch 3/10
50/50 [==============================] - 88s 2s/step - loss: 12.0338 - loss_ce: 4.5101 - loss_l1: 5.3720 - loss_giou: 2.1516
Epoch 4/10
50/50 [==============================] - 102s 2s/step - loss: 11.9497 - loss_ce: 4.5096 - loss_l1: 5.3074 - loss_giou: 2.1327
Epoch 5/10
50/50 [==============================] - 95s 2s/step - loss: 11.8448 - loss_ce: 4.5088 - loss_l1: 5.2172 - loss_giou: 2.1189
Epoch 6/10
50/50 [==============================] - 91s 2s/step - loss: 11.7592 - loss_ce: 4.5089 - loss_l1: 5.1461 - loss_giou: 2.1042
Epoch 7/10
50/50 [==============================] - 92s 2s/step - loss: 11.7861 - loss_ce: 4.5089 - loss_l1: 5.1637 - loss_giou: 2.1135
Epoch 8/10
50/50 [==============================] - 100s 2s/step - loss: 11.7100 - loss_ce: 4.5090 - loss_l1: 5.1021 - loss_giou: 2.0990
Epoch 9/10
50/50 [==============================] - 91s 2s/step - loss: 11.8219 - loss_ce: 4.5091 - loss_l1: 5.1911 - loss_giou: 2.1217
Epoch 10/10
50/50 [==============================] - 84s 2s/step - loss: 11.7200 - loss_ce: 4.5095 - loss_l1: 5.0921 - loss_giou: 2.1184

"""

# 2 GPU:
"""
10 epochs (placeholder)
25/25 [==============================] - 109s 4s/step - loss: 4.5211
Epoch 2/10
25/25 [==============================] - 65s 3s/step - loss: 4.5197
Epoch 3/10
25/25 [==============================] - 51s 2s/step - loss: 4.5183
Epoch 4/10
25/25 [==============================] - 63s 3s/step - loss: 4.5173
Epoch 5/10
25/25 [==============================] - 55s 2s/step - loss: 4.5160
Epoch 6/10
25/25 [==============================] - 55s 2s/step - loss: 4.5161
Epoch 7/10
25/25 [==============================] - 55s 2s/step - loss: 4.5161
Epoch 8/10
25/25 [==============================] - 59s 2s/step - loss: 4.5158
Epoch 9/10
25/25 [==============================] - 53s 2s/step - loss: 4.5166
Epoch 10/10
25/25 [==============================] - 46s 2s/step - loss: 4.5160

TODO: 10 epochs (placeholder, no prefetch)

10 epochs (hungarian)
25/25 [==============================] - 129s 5s/step - loss: 11.8861
Epoch 2/10
25/25 [==============================] - 81s 3s/step - loss: 11.6242
Epoch 3/10
25/25 [==============================] - 68s 3s/step - loss: 12.0430
Epoch 4/10
25/25 [==============================] - 81s 3s/step - loss: 11.9534
Epoch 5/10
25/25 [==============================] - 71s 3s/step - loss: 11.8452
Epoch 6/10
25/25 [==============================] - 72s 3s/step - loss: 11.7589
Epoch 7/10
25/25 [==============================] - 74s 3s/step - loss: 11.7853
Epoch 8/10
25/25 [==============================] - 76s 3s/step - loss: 11.7085
Epoch 9/10
25/25 [==============================] - 69s 3s/step - loss: 11.8199
Epoch 10/10
25/25 [==============================] - 61s 2s/step - loss: 11.7179

TODO: 10 epochs (hungarian, no prefetch)
"""
