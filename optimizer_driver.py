import tensorflow as tf
import tensorflow_addons as tfa

from chambers.optimizers import LearningRateMultiplier
from models import build_detr_resnet50

# %%
optw = tfa.optimizers.AdamW(weight_decay=1e-4,
                           learning_rate=1e-4,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-8,
                           clipnorm=0.1,
                           amsgrad=False
                           )

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5,
                                                             decay_steps=100,
                                                             decay_rate=0.1,
                                                             staircase=False)

multipliers = {"backbone": 0.1}
opt = LearningRateMultiplier(optimizer=optw,
                             lr_multipliers=multipliers,
                             )

#%%


#%%
print(optw.lr.numpy(), opt.lr.numpy())
opt.lr = 1
print(optw.lr.numpy(), opt.lr.numpy())
opt._optimizer.lr = 2
print(optw.lr.numpy(), opt.lr.numpy())
opt._optimizer._set_hyper("learning_rate", 3)
print(optw.lr.numpy(), opt.lr.numpy())

# %%
detr = build_detr_resnet50(num_classes=91,
                           num_queries=100,
                           mask_value=-1.,
                           return_decode_sequence=False)
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

# %%
params = detr.trainable_variables  # .backbone.trainable_variables
ml = opt._get_params_multipliers(params)

# %%
# for v1, (v2, lr) in zip(params, mult_lr_params.items()):
#     n1 = v1.name
#     n2 = v2.deref().name
#     print(n1 == n2, n1, n2, lr)
