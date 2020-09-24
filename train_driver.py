import os

import tensorflow as tf
import tensorflow_addons as tfa

from chambers.callbacks import DETRLossDiffLogger, HungarianLossLogger, DETRPredImageTensorboard, GroupedTensorBoard
from chambers.losses import HungarianLoss
from chambers.models.detr import DETR, load_detr
from chambers.optimizers import LearningRateMultiplier
from chambers.utils.utils import timestamp_now
from data.tf_datasets import load_coco_tf, CLASSES_TF

# model_path = "outputs/2020-09-24_17-01-41/checkpoints/model-init.h5"
model_path = None

# %% strategy
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
        detr = load_detr(model_path)
    else:
        print("Initializing model...")
        detr = build_and_compile_detr()

detr.summary()

# %% train

# set training configuration
print("\n### TRAINING ###")
EPOCHS = 2  # 150
# n_train = 200
# n_val = 100
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
                                       ),
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
* Test loading of model
* Test model on 8 GPUs
    * set caching for datasets
    * Set EPOCHS = 150
"""

""" 6 GPU
18/18 [==============================] - 157s 9s/step - loss: 86.6874
Epoch 2/10
18/18 [==============================] - 37s 2s/step - loss: 58.2329
Epoch 3/10
18/18 [==============================] - 35s 2s/step - loss: 53.1096
Epoch 4/10
18/18 [==============================] - 33s 2s/step - loss: 51.3836
Epoch 5/10
18/18 [==============================] - 35s 2s/step - loss: 50.1125
"""
""" 7 GPU
18/18 [==============================] - 171s 9s/step - loss: 86.6874
Epoch 2/10
18/18 [==============================] - 30s 2s/step - loss: 58.2329
Epoch 3/10
18/18 [==============================] - 30s 2s/step - loss: 53.1096
Epoch 4/10
18/18 [==============================] - 30s 2s/step - loss: 51.3836
Epoch 5/10
18/18 [==============================] - 30s 2s/step - loss: 50.1125
"""
""" 8 GPU
13/13 [==============================] - 169s 13s/step - loss: 99.4983
Epoch 2/10
13/13 [==============================] - 27s 2s/step - loss: 61.2178
Epoch 3/10
13/13 [==============================] - 26s 2s/step - loss: 54.2451
Epoch 4/10
13/13 [==============================] - 27s 2s/step - loss: 52.1206
"""

""" COCO 91
Epoch 1/5
2020-09-23 19:55:43.120492: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-09-23 19:55:43.509665: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
66/66 [==============================] - 135s 2s/step - loss: 73.6972 - val_loss: 85.7333 - val_loss_ce: 3.5864 - val_loss_l1: 5.4201 - val_loss_giou: 3.5038 - val_loss_ce_0: 3.5892 - val_loss_l1_0: 7.4665 - val_loss_giou_0: 3.5365 - val_loss_ce_1: 3.5418 - val_loss_l1_1: 9.0946 - val_loss_giou_1: 3.5060 - val_loss_ce_2: 3.6339 - val_loss_l1_2: 7.6095 - val_loss_giou_2: 3.6349 - val_loss_ce_3: 3.6477 - val_loss_l1_3: 7.9458 - val_loss_giou_3: 2.5667 - val_loss_ce_4: 3.8262 - val_loss_l1_4: 6.3148 - val_loss_giou_4: 3.3089 - train_diff: 48.2684 - test_diff: 57.3350
Epoch 2/5
66/66 [==============================] - 110s 2s/step - loss: 53.0950 - val_loss: 63.5761 - val_loss_ce: 3.3710 - val_loss_l1: 4.6353 - val_loss_giou: 2.2734 - val_loss_ce_0: 3.3388 - val_loss_l1_0: 4.7615 - val_loss_giou_0: 2.2537 - val_loss_ce_1: 3.3255 - val_loss_l1_1: 4.3456 - val_loss_giou_1: 2.4797 - val_loss_ce_2: 3.3922 - val_loss_l1_2: 4.6213 - val_loss_giou_2: 2.3083 - val_loss_ce_3: 3.3898 - val_loss_l1_3: 4.5641 - val_loss_giou_3: 2.4860 - val_loss_ce_4: 3.5285 - val_loss_l1_4: 5.0691 - val_loss_giou_4: 3.4324 - train_diff: 32.3907 - test_diff: 44.4805
Epoch 3/5
66/66 [==============================] - 106s 2s/step - loss: 50.1554 - val_loss: 61.4871 - val_loss_ce: 3.3355 - val_loss_l1: 4.8263 - val_loss_giou: 2.2223 - val_loss_ce_0: 3.3019 - val_loss_l1_0: 4.4663 - val_loss_giou_0: 2.3312 - val_loss_ce_1: 3.2895 - val_loss_l1_1: 4.3478 - val_loss_giou_1: 2.4330 - val_loss_ce_2: 3.3535 - val_loss_l1_2: 4.5591 - val_loss_giou_2: 2.4389 - val_loss_ce_3: 3.3490 - val_loss_l1_3: 4.6252 - val_loss_giou_3: 2.3877 - val_loss_ce_4: 3.4801 - val_loss_l1_4: 4.6103 - val_loss_giou_4: 2.1294 - train_diff: 32.4622 - test_diff: 44.5629
Epoch 4/5
66/66 [==============================] - 113s 2s/step - loss: 47.8907 - val_loss: 60.5322 - val_loss_ce: 3.3322 - val_loss_l1: 4.4163 - val_loss_giou: 2.3041 - val_loss_ce_0: 3.3025 - val_loss_l1_0: 4.4852 - val_loss_giou_0: 2.2406 - val_loss_ce_1: 3.2863 - val_loss_l1_1: 4.3762 - val_loss_giou_1: 2.3816 - val_loss_ce_2: 3.3494 - val_loss_l1_2: 4.3240 - val_loss_giou_2: 2.4141 - val_loss_ce_3: 3.3454 - val_loss_l1_3: 4.3778 - val_loss_giou_3: 2.3433 - val_loss_ce_4: 3.4737 - val_loss_l1_4: 4.3844 - val_loss_giou_4: 2.3950 - train_diff: 31.9598 - test_diff: 45.3513
Epoch 5/5
66/66 [==============================] - 106s 2s/step - loss: 47.8675 - val_loss: 60.3014 - val_loss_ce: 3.3327 - val_loss_l1: 4.3626 - val_loss_giou: 2.3735 - val_loss_ce_0: 3.3047 - val_loss_l1_0: 4.4001 - val_loss_giou_0: 2.3041 - val_loss_ce_1: 3.2861 - val_loss_l1_1: 4.3724 - val_loss_giou_1: 2.3087 - val_loss_ce_2: 3.3476 - val_loss_l1_2: 4.3707 - val_loss_giou_2: 2.2954 - val_loss_ce_3: 3.3448 - val_loss_l1_3: 4.3912 - val_loss_giou_3: 2.3101 - val_loss_ce_4: 3.4708 - val_loss_l1_4: 4.3303 - val_loss_giou_4: 2.3957 - train_diff: 32.8998 - test_diff: 45.9362

COCO 80
Epoch 1/5
2020-09-23 20:37:03.879364: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-09-23 20:37:04.460826: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
66/66 [==============================] - 128s 2s/step - loss: 72.6211 - val_loss: 81.7943 - val_loss_ce: 3.6545 - val_loss_l1: 7.7759 - val_loss_giou: 3.6723 - val_loss_ce_0: 3.6185 - val_loss_l1_0: 4.5250 - val_loss_giou_0: 2.5362 - val_loss_ce_1: 3.7370 - val_loss_l1_1: 6.6660 - val_loss_giou_1: 3.2471 - val_loss_ce_2: 3.6172 - val_loss_l1_2: 6.0165 - val_loss_giou_2: 2.5517 - val_loss_ce_3: 3.5320 - val_loss_l1_3: 7.6195 - val_loss_giou_3: 3.3181 - val_loss_ce_4: 3.6503 - val_loss_l1_4: 8.5809 - val_loss_giou_4: 3.4754 - train_diff: 47.1922 - test_diff: 53.3959
Epoch 2/5
66/66 [==============================] - 109s 2s/step - loss: 52.0335 - val_loss: 65.9819 - val_loss_ce: 3.4479 - val_loss_l1: 4.6987 - val_loss_giou: 2.4452 - val_loss_ce_0: 3.4183 - val_loss_l1_0: 4.8866 - val_loss_giou_0: 2.8173 - val_loss_ce_1: 3.5178 - val_loss_l1_1: 4.7048 - val_loss_giou_1: 2.4138 - val_loss_ce_2: 3.3554 - val_loss_l1_2: 4.9410 - val_loss_giou_2: 2.3095 - val_loss_ce_3: 3.3345 - val_loss_l1_3: 5.3597 - val_loss_giou_3: 2.6008 - val_loss_ce_4: 3.4068 - val_loss_l1_4: 5.7898 - val_loss_giou_4: 2.5341 - train_diff: 31.3291 - test_diff: 46.8863
Epoch 3/5
66/66 [==============================] - 106s 2s/step - loss: 47.5053 - val_loss: 63.3906 - val_loss_ce: 3.4172 - val_loss_l1: 5.0784 - val_loss_giou: 2.4578 - val_loss_ce_0: 3.3905 - val_loss_l1_0: 4.5234 - val_loss_giou_0: 2.4402 - val_loss_ce_1: 3.4815 - val_loss_l1_1: 4.6718 - val_loss_giou_1: 2.4627 - val_loss_ce_2: 3.3190 - val_loss_l1_2: 4.7211 - val_loss_giou_2: 2.3471 - val_loss_ce_3: 3.3002 - val_loss_l1_3: 4.5658 - val_loss_giou_3: 2.5014 - val_loss_ce_4: 3.3751 - val_loss_l1_4: 4.7362 - val_loss_giou_4: 2.6012 - train_diff: 29.8121 - test_diff: 46.4664
Epoch 4/5
66/66 [==============================] - 109s 2s/step - loss: 47.3689 - val_loss: 61.8971 - val_loss_ce: 3.4138 - val_loss_l1: 4.5793 - val_loss_giou: 2.2992 - val_loss_ce_0: 3.3902 - val_loss_l1_0: 4.6353 - val_loss_giou_0: 2.2906 - val_loss_ce_1: 3.4760 - val_loss_l1_1: 4.5851 - val_loss_giou_1: 2.3810 - val_loss_ce_2: 3.3173 - val_loss_l1_2: 4.6101 - val_loss_giou_2: 2.3306 - val_loss_ce_3: 3.2980 - val_loss_l1_3: 4.6040 - val_loss_giou_3: 2.4212 - val_loss_ce_4: 3.3724 - val_loss_l1_4: 4.5886 - val_loss_giou_4: 2.3047 - train_diff: 31.4379 - test_diff: 46.7163
Epoch 5/5
66/66 [==============================] - 105s 2s/step - loss: 45.7577 - val_loss: 61.9342 - val_loss_ce: 3.4128 - val_loss_l1: 4.6068 - val_loss_giou: 2.2957 - val_loss_ce_0: 3.3912 - val_loss_l1_0: 4.5765 - val_loss_giou_0: 2.3029 - val_loss_ce_1: 3.4728 - val_loss_l1_1: 4.4848 - val_loss_giou_1: 2.4970 - val_loss_ce_2: 3.3177 - val_loss_l1_2: 4.5825 - val_loss_giou_2: 2.3485 - val_loss_ce_3: 3.2984 - val_loss_l1_3: 4.6520 - val_loss_giou_3: 2.3443 - val_loss_ce_4: 3.3711 - val_loss_l1_4: 4.7385 - val_loss_giou_4: 2.2405 - train_diff: 30.7901 - test_diff: 47.5690

"""
