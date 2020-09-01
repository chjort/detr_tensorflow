import tensorflow as tf
import tensorflow_addons as tfa

from chambers.callbacks import HungarianLossLogger
from chambers.losses import HungarianLoss, pairwise_softmax, pairwise_l1, pairwise_giou
from chambers.optimizers import LearningRateMultiplier
from models import build_detr_resnet50
from tf_datasets import load_coco

# %%
BATCH_SIZE = 4
# dataset, N = load_coco_tf("train", BATCH_SIZE)
dataset, N = load_coco("/datadrive/crr/datasets/coco", "train", BATCH_SIZE)
# dataset, N = load_coco("/home/ch/datasets/coco", "train", BATCH_SIZE)

dataset = dataset.prefetch(-1)

EPOCHS = 3  # 150
STEPS_PER_EPOCH = 50  # N

# %%
decode_sequence = False
detr = build_detr_resnet50(num_classes=91,
                           num_queries=100,
                           mask_value=-1.,
                           return_decode_sequence=decode_sequence)
detr.build()
# detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

hungarian = HungarianLoss(lsa_losses=[pairwise_softmax, pairwise_l1, pairwise_giou],
                          lsa_loss_weights=[1, 5, 2],
                          mask_value=-1.,
                          sequence_input=decode_sequence)

# %% COMPILE

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
                           clipnorm=0.1,
                           amsgrad=False,
                           )
opt = LearningRateMultiplier(opt, lr_multipliers={"backbone": 0.1})

detr.compile(optimizer=opt,
             loss=hungarian,
             )

# %% TRAIN
# ssh -L 6006:127.0.0.1:6006 crr@40.68.160.55
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="tb_logs", write_graph=False, update_freq="epoch", profile_batch=0)
history = detr.fit(dataset,
                   epochs=EPOCHS,
                   steps_per_epoch=STEPS_PER_EPOCH,
                   callbacks=[HungarianLossLogger(), tensorboard]
                   )
