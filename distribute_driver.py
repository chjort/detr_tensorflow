import tensorflow as tf
import tensorflow_addons as tfa

from chambers.callbacks import HungarianLossLogger
from chambers.losses import HungarianLoss, pairwise_softmax, pairwise_l1, pairwise_giou
from chambers.optimizers import LearningRateMultiplier
from models import build_detr_resnet50
from tf_datasets import load_coco
import tensorflow_datasets as tfds

# %% Multi devices
physical_gpus = tf.config.experimental.list_physical_devices("GPU")
print("Number of physical GPUs:", len(physical_gpus))

tf.config.experimental.set_virtual_device_configuration(device=physical_gpus[0],
                                                        logical_devices=[
                                                            tf.config.experimental.VirtualDeviceConfiguration(10240),
                                                            tf.config.experimental.VirtualDeviceConfiguration(10240)
                                                        ]
                                                        )
logical_gpus = tf.config.experimental.list_logical_devices("GPU")
print("Number of logical GPUs:", len(logical_gpus))

#%% Strategy
strategy = tf.distribute.MirroredStrategy()
print("Number of devices in strategy:", strategy.num_replicas_in_sync)

# %% Data
dataset, info = tfds.load("mnist",
                          data_dir="/datadrive/crr/tensorflow_datasets",
                          with_info=True,
                          as_supervised=True
                          )
x_train = dataset["train"]
x_test = dataset["test"]

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image, label

x_train = x_train.map(scale).cache().repeat(-1).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
x_test = x_test.map(scale).batch(GLOBAL_BATCH_SIZE)


next(iter(x_train))

#%% Model
with strategy.scope():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.models.Model([inputs], [x])

    lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 4, 0.1)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[metric]
              )

#%%
model.fit(x_train, validation_data=x_test, epochs=12, steps_per_epoch=60000)
# 1 device: 151, 149
# 2 device: 261