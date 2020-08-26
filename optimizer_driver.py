import os
import tensorflow_addons as tfa
from chambers.optimizers import AdamWLearningRateMultiplier
from models import build_detr_resnet50

# %%
# opt = tfa.optimizers.AdamW(weight_decay=1e-4,
#                            learning_rate=1e-4,  # TODO: 1e-4 for transformer, and 1e-5 for backbone
#                            beta_1=0.9,
#                            beta_2=0.999,
#                            epsilon=1e-8,
#                            amsgrad=False
#                            )

# %%
# opt = LearningRateMultiplier(tfa.optimizers.AdamW,
#                              lr_multipliers={"backbone": 0.1},
#                              weight_decay=1e-4,
#                              learning_rate=1e-4,
#                              beta_1=0.9,
#                              beta_2=0.999,
#                              epsilon=1e-8,
#                              amsgrad=False
#                              )

# %%
multipliers = {"backbone": 0.1, "backbone/0/body/layer4": 5, "backbone/0/body/layer4/1": 2}
opt = AdamWLearningRateMultiplier(weight_decay=1e-4,
                                  learning_rate=1e-4,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-8,
                                  amsgrad=False,
                                  lr_multipliers=multipliers,
                                  )

# %%
detr = build_detr_resnet50(num_classes=91,
                           num_queries=100,
                           mask_value=-1.,
                           return_decode_sequence=False)
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

# %%
params = detr.trainable_variables#.backbone.trainable_variables
ml = opt._get_params_multipliers(params)

# %%
# for v1, (v2, lr) in zip(params, mult_lr_params.items()):
#     n1 = v1.name
#     n2 = v2.deref().name
#     print(n1 == n2, n1, n2, lr)
