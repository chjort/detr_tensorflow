import os

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
import tensorflow as tf
import tensorflow_addons as tfa

from chambers.callbacks import DETRLossDiffLogger, HungarianLossLogger, DETRPredImageTensorboard, GroupedTensorBoard
from chambers.losses.hungarian import HungarianLoss
from chambers.models.detr import DETR
from chambers.optimizers import LearningRateMultiplier
from chambers.utils.utils import timestamp_now
from data.tf_datasets import load_coco_tf, CLASSES_TF, load_coco, CLASSES_COCO

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
# train_dataset, n_train = load_coco_tf("train", GLOBAL_BATCH_SIZE, "/datadrive/crr/tensorflow_datasets")
# val_dataset, n_val = load_coco_tf("validation", GLOBAL_BATCH_SIZE, "/datadrive/crr/tensorflow_datasets")
train_dataset, n_train = load_coco("/datadrive/crr/datasets/coco", "train", GLOBAL_BATCH_SIZE)
val_dataset, n_val = load_coco("/datadrive/crr/datasets/coco", "val", GLOBAL_BATCH_SIZE)

train_dataset = train_dataset.prefetch(-1)
val_dataset = val_dataset.prefetch(-1)

CLASSES = CLASSES_COCO
# CLASSES = CLASSES_TF
N_CLASSES = len(CLASSES)
print("Number of training samples:", n_train)
print("Number of validation samples:", n_val)
print("Number of classes:", N_CLASSES)

# %%
# from chambers.utils.utils import plot_results
# from chambers.utils.image import resnet_imagenet_denormalize
# from chambers.utils.masking import remove_padding_image, remove_padding_box
# from chambers.augmentations import resize, box_denormalize_yxyx, box_normalize_yxyx, flip_left_right
#
# it = iter(train_dataset)
# x, y = next(it)
# x = x[0]
# y = y[0]
#
# y = y[:, :-1]
# # xp = remove_padding_image(x, -1.)
# xp = x
# xp = resnet_imagenet_denormalize(xp)
# yp = remove_padding_box(y, -1.)
#
# print(xp.shape)
# plot_results(xp, yp)
#
# yp = box_denormalize_yxyx(yp, xp)
# aug fn
# yp = box_normalize_yxyx(yp, xp)
# plot_results(xp, yp)

# %% building model
print("\n### BUILDING MODEL ###")


def build_and_compile_detr():
    return_decode_sequence = True
    mask_value = -1.
    detr = DETR(input_shape=(None, None, 3),
    # detr=DETR(input_shape=(768, 768, 3),
                n_classes=N_CLASSES,
                n_object_queries=100,
                embed_dim=256,
                num_heads=8,
                dim_feedforward=2048,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dropout_rate=0.1,
                return_decode_sequence=return_decode_sequence,
                mask_value=mask_value
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

    # TODO: Could the problem be that learning rates are reduced for every step???
    #   Causing the model to not learn anything after around ~1200 steps with a loss of ~44.
    #   After this point the loss increases as the model is shown new examples but does not learn.
    opt = LearningRateMultiplier(opt,
                                 lr_multipliers={var.name: 0.1 for var in detr.get_layer("resnet50").variables}
                                 # lr_multipliers={"resnet50": 0.1}
                                 )

    hungarian = HungarianLoss(n_classes=N_CLASSES,
                              loss_weights=[1, 5, 2],
                              lsa_loss_weights=[1, 5, 2],
                              mask_value=mask_value,
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
EPOCHS = 10  # 150
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
                             label_names=CLASSES
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

# Tensorboard: # ssh -L 6006:127.0.0.1:6006 crr@51.144.79.65
