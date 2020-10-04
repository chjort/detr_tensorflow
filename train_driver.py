import os

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"  # TODO: Try on 8 GPUs with this setting
import tensorflow as tf
import tensorflow_addons as tfa

from chambers.callbacks import DETRLossDiffLogger, HungarianLossLogger, DETRPredImageTensorboard, GroupedTensorBoard
from chambers.losses import HungarianLoss
from chambers.models.detr import DETR, load_detr
from chambers.optimizers import LearningRateMultiplier
from chambers.utils.utils import timestamp_now
from data.tf_datasets import load_coco_tf, CLASSES_TF


# def loss_placeholder(y_true, y_pred):
#     y_true_labels = y_true[..., -1]  # [1]
#     y_pred_logits = y_pred[..., 4:]  # [1]
#
#     y_pred = y_pred_logits[:, -1, :1]
#     y_true = y_true_labels[:, :1]
#
#     return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


# model_path = "outputs/2020-10-04_19-19-45/checkpoints/model-init.h5"
model_path = None

# %% strategy
# Using 8 GPU will give "WARNING:tensorflow:Large unrolled loop detected"
#   at tensorflow/python/autograph/operators/control_flow.py, line 817, in _verify_inefficient_unroll
#   This warning should simply be ignored.

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3",
                                                   "/gpu:4", "/gpu:5", "/gpu:6",
                                                   "/gpu:7"
                                                   ])
# strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

# %% loading data
BATCH_SIZE_PER_REPLICA = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print("\n### LOADING DATA ###")
train_dataset, n_train = load_coco_tf("train", GLOBAL_BATCH_SIZE, "/datadrive/crr/tensorflow_datasets")
val_dataset, n_val = load_coco_tf("validation", GLOBAL_BATCH_SIZE, "/datadrive/crr/tensorflow_datasets")

train_dataset = train_dataset.prefetch(-1)
val_dataset = val_dataset.prefetch(-1)

N_CLASSES = len(CLASSES_TF)
print("Number of training samples:", n_train)
print("Number of validation samples:", n_val)
print("Number of classes:", N_CLASSES)

# %% building model
print("\n### BUILDING MODEL ###")


def build_and_compile_detr():
    return_decode_sequence = True
    detr = DETR(input_shape=(None, None, 3),
                n_classes=N_CLASSES,
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
                               # clipnorm=0.1,  # 'clipnorm' argument not compatible with Multi-GPU training
                               amsgrad=False,
                               )

    opt = LearningRateMultiplier(opt, lr_multipliers={"resnet50": 0.1})

    hungarian = HungarianLoss(n_classes=N_CLASSES,
                              loss_weights=[1, 5, 2],
                              lsa_loss_weights=[1, 5, 2],
                              mask_value=-1.,
                              sequence_input=return_decode_sequence)
    detr.compile(optimizer=opt,
                 loss=hungarian,
                 )
    return detr


# build model
with strategy.scope():
    if model_path is not None:
        print("Loading model:", model_path)
        detr = tf.keras.models.load_model(model_path)
    else:
        print("Initializing model...")
        detr = build_and_compile_detr()

detr.summary()

# %% train

# set training configuration
print("\n### TRAINING ###")
EPOCHS = 50  # 150
# n_train = 416
# n_val = 416
STEPS_PER_EPOCH = n_train // GLOBAL_BATCH_SIZE
VAL_STEPS_PER_EPOCH = n_val // GLOBAL_BATCH_SIZE

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

# create callbacks
callbacks = [
    HungarianLossLogger(val_dataset, steps=VAL_STEPS_PER_EPOCH),
    DETRLossDiffLogger("samples/fb_log.txt"),
    tf.keras.callbacks.CSVLogger(csv_path),
    GroupedTensorBoard(loss_groups=["val_loss_ce", "val_loss_l1", "val_loss_giou"],
                       writer_prefixes="decode_layer",
                       log_dir=tensorboard_path,
                       write_graph=True,
                       update_freq="epoch",
                       profile_batch=0),
    DETRPredImageTensorboard(log_dir=tensorboard_path,
                             min_prob=(0.0, 0.1, 0.5, 0.7),
                             image_files=["samples/sample0.jpg",
                                          "samples/sample1.png",
                                          "samples/sample2.jpg"
                                          ],
                             label_names=CLASSES_TF
                             ),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                       monitor="val_loss",
                                       save_best_only=False,
                                       save_weights_only=False
                                       )
]

# fit
detr.save(os.path.join(checkpoint_dir, "model-init.h5"))  # save initialization
history = detr.fit(train_dataset,
                   epochs=EPOCHS,
                   steps_per_epoch=STEPS_PER_EPOCH,
                   callbacks=callbacks
                   )

# Tensorboard: # ssh -L 6006:127.0.0.1:6006 crr@40.68.160.55

""" TODO:
* Test model on 8 GPUs
    * Set EPOCHS = 150
"""

""" 4 GPU (bs 416, decode_sequence=True)
26/26 [==============================] - 97s 4s/step - loss: 86.8079
Epoch 2/10
26/26 [==============================] - 27s 1s/step - loss: 52.3061
Epoch 3/10
26/26 [==============================] - 28s 1s/step - loss: 48.4626
Epoch 4/10
26/26 [==============================] - 28s 1s/step - loss: 46.8509
Epoch 5/10
26/26 [==============================] - 28s 1s/step - loss: 44.9585

8 GPU (bs 416, decode_sequence=True)
13/13 [==============================] - 171s 13s/step - loss: 99.2107
Epoch 2/10
13/13 [==============================] - 21s 2s/step - loss: 57.3480
Epoch 3/10
13/13 [==============================] - 22s 2s/step - loss: 52.3514
Epoch 4/10
13/13 [==============================] - 21s 2s/step - loss: 50.0755
Epoch 5/10
13/13 [==============================] - 23s 2s/step - loss: 49.0004

8 GPU (bs 416, decode_sequence=False)
13/13 [==============================] - 157s 12s/step - loss: 21.0014
Epoch 2/10
13/13 [==============================] - 11s 876ms/step - loss: 9.3689
Epoch 3/10
13/13 [==============================] - 11s 851ms/step - loss: 8.2976
Epoch 4/10
13/13 [==============================] - 11s 877ms/step - loss: 8.0504
Epoch 5/10
13/13 [==============================] - 13s 964ms/step - loss: 7.9375
"""