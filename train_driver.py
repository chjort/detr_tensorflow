import os

import tensorflow as tf
import tensorflow_addons as tfa

from chambers.callbacks import DETR_FB_Loss_Diff, HungarianLossLogger
from chambers.losses import HungarianLoss
from chambers.models.detr import DETR, load_detr
from chambers.optimizers import LearningRateMultiplier
from chambers.utils.utils import timestamp_now
from data.tf_datasets import load_coco

# model_path = "outputs/2020-09-14_20:58:52/model-epoch2.h5"
model_path = None

# %% strategy
strategy = tf.distribute.MirroredStrategy()

# %%
BATCH_SIZE_PER_REPLICA = 3
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print("\n### LOADING DATA ###")
train_dataset, N_train = load_coco("/datadrive/crr/datasets/coco", "train", GLOBAL_BATCH_SIZE)
val_dataset, N_val = load_coco("/datadrive/crr/datasets/coco", "val", GLOBAL_BATCH_SIZE)

train_dataset = train_dataset.prefetch(-1)

# %%
print("\n### BUILDING MODEL ###")


def build_and_compile_detr():
    return_decode_sequence = True
    detr = DETR(input_shape=(None, None, 3),
                n_classes=91,
                n_object_queries=100,
                embed_dim=256,
                num_heads=8,
                dim_feedforward=2048,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dropout_rate=0.1,
                return_decode_sequence=return_decode_sequence,
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

    var_lr_mult = {var.name: 0.1 for var in detr.get_layer("resnet50").variables}
    opt = LearningRateMultiplier(opt, lr_multipliers=var_lr_mult)

    hungarian = HungarianLoss(loss_weights=[1, 5, 2],
                              lsa_loss_weights=[1, 5, 2],
                              mask_value=-1.,
                              sequence_input=return_decode_sequence)
    detr.compile(optimizer=opt,
                 loss=hungarian,
                 )
    return detr


with strategy.scope():
    if model_path is not None:
        print("Loading model:", model_path)
        detr = load_detr(model_path)
    else:
        print("Initializing model.")
        detr = build_and_compile_detr()

# %% TRAIN
print("\n### TRAINING ###")
EPOCHS = 10  # 150
N_train = 200
N_val = 100
STEPS_PER_EPOCH = N_train // GLOBAL_BATCH_SIZE
VAL_STEPS_PER_EPOCH = N_val // GLOBAL_BATCH_SIZE

print("Number of GPUs for training:", strategy.num_replicas_in_sync)
print("Global batch size: {}. Per GPU batch size: {}".format(GLOBAL_BATCH_SIZE, BATCH_SIZE_PER_REPLICA))

print("Training steps per epoch:", STEPS_PER_EPOCH)
print("Validation steps per epoch:", VAL_STEPS_PER_EPOCH)

# make output folders and paths
model_dir = os.path.join("outputs", timestamp_now())
checkpoint_dir = os.path.join(model_dir, "checkpoints")
log_dir = os.path.join(model_dir, "logs")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "model-epoch{epoch}.h5")
csv_path = os.path.join(log_dir, "logs.csv")
tensorboard_path = os.path.join(log_dir, "tb")

detr.save(os.path.join(checkpoint_dir, "model-init.h5"))  # save initial weights
history = detr.fit(train_dataset,
                   epochs=EPOCHS,
                   steps_per_epoch=STEPS_PER_EPOCH,
                   callbacks=[
                       HungarianLossLogger(val_dataset.take(VAL_STEPS_PER_EPOCH)),
                       DETR_FB_Loss_Diff("samples/fb_log.txt"),
                       tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                          monitor="val_loss",
                                                          save_best_only=False,
                                                          save_weights_only=False
                                                          ),
                       # ssh -L 6006:127.0.0.1:6006 crr@40.68.160.55
                       tf.keras.callbacks.CSVLogger(csv_path),
                       tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, write_graph=True,
                                                      update_freq="epoch", profile_batch=0)
                   ]
                   )

""" TODO:
* Log output prediction images into Tensorboard
* Group sublosses for decode layers in Tensorboard
* Test model on 8 GPUs
* Set device batch size to 4 on Tesla V100 32GB
* Set EPOCHS = 150
"""
