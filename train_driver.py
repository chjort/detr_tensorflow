import os

import tensorflow as tf
import tensorflow_addons as tfa

from chambers.losses import HungarianLoss, pairwise_softmax, pairwise_l1, pairwise_giou
from chambers.models.detr import DETR, load_detr
from chambers.optimizers import LearningRateMultiplier
from chambers.utils.utils import timestamp_now
from tf_datasets import load_coco


def save_all_weights(model, path):
    folder, fname = os.path.split(path)
    os.makedirs(folder, exist_ok=True)
    model.save_weights(path)
    tf.saved_model.save(model.optimizer, "{}-opt.{}".format(*path.split(".")[-2:]))


def loss_placeholder(y_true, y_pred):
    y_true_labels = y_true[..., -1]  # [1]
    y_pred_logits = y_pred[..., 4:]  # [1]
    y_pred = y_pred_logits[:, :1, :]
    y_true = tf.one_hot(tf.cast(y_true_labels, tf.int32), depth=92)[:, :1, :]
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)


# model_path = "outputs/2020-09-14_20:58:52/model-epoch2.h5"
model_path = None

# %% strategy
# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

# %%
BATCH_SIZE_PER_REPLICA = 3
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# train_dataset, N_train = load_coco_tf("train", GLOBAL_BATCH_SIZE)
# val_dataset, N_val = load_coco_tf("val", GLOBAL_BATCH_SIZE)
train_dataset, N_train = load_coco("/datadrive/crr/datasets/coco", "train", GLOBAL_BATCH_SIZE)
val_dataset, N_val = load_coco("/datadrive/crr/datasets/coco", "val", GLOBAL_BATCH_SIZE)

train_dataset = train_dataset.prefetch(-1)

# %%
with strategy.scope():
    if model_path is not None:
        print("Loading model:", model_path)
        detr = load_detr(model_path)
    else:
        decode_sequence = False
        detr = DETR(input_shape=(None, None, 3),
                    n_classes=91,
                    n_object_queries=100,
                    embed_dim=256,
                    num_heads=8,
                    dim_feedforward=2048,
                    num_encoder_layers=6,
                    num_decoder_layers=6,
                    dropout_rate=0.1,
                    return_decode_sequence=decode_sequence,
                    mask_value=-1.
                    )

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
                     )

# %% TRAIN
EPOCHS = 5  # 150
N_train = 200
N_val = 100
STEPS_PER_EPOCH = N_train / GLOBAL_BATCH_SIZE
VAL_STEPS = N_val / GLOBAL_BATCH_SIZE

print("Number of devices in strategy:", strategy.num_replicas_in_sync)
print("Global batch size: {}. Per device batch size: {}".format(GLOBAL_BATCH_SIZE, BATCH_SIZE_PER_REPLICA))

# ssh -L 6006:127.0.0.1:6006 crr@40.68.160.55
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="tb_logs", write_graph=True, update_freq="epoch", profile_batch=0)

model_dir = os.path.join("outputs", timestamp_now())
os.makedirs(model_dir, exist_ok=True)
model_file = os.path.join(model_dir, "model-epoch{epoch}.h5")
history = detr.fit(train_dataset,
                   validation_data=val_dataset,
                   epochs=EPOCHS,
                   steps_per_epoch=STEPS_PER_EPOCH,
                   validation_steps=VAL_STEPS,
                   callbacks=[
                       # HungarianLossLogger(),
                       tf.keras.callbacks.ModelCheckpoint(filepath=model_file,
                                                          monitor="val_loss",
                                                          save_best_only=False,
                                                          save_weights_only=False
                                                          ),
                       tensorboard
                   ]
                   )
