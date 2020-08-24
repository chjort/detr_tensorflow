from chambers.losses import HungarianLoss, pairwise_softmax, pairwise_l1, pairwise_giou
from chambers.optimizers import AdamWLearningRateMultiplier
import tensorflow_addons as tfa
from models import build_detr_resnet50
from tf_datasets import load_coco

# %%
BATCH_SIZE = 2
# dataset, N = load_coco_tf("train", BATCH_SIZE)
dataset, N = load_coco("/datadrive/crr/datasets/coco", "train", BATCH_SIZE)
# dataset, N = load_coco("/home/ch/datasets/coco", "train", BATCH_SIZE)

# %%
decode_sequence = False
detr = build_detr_resnet50(num_classes=91,
                           num_queries=100,
                           mask_value=-1.,
                           return_decode_sequence=decode_sequence)
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

hungarian = HungarianLoss(lsa_losses=[pairwise_softmax, pairwise_l1, pairwise_giou],
                          lsa_loss_weights=[1, 5, 2],
                          mask_value=-1.,
                          sequence_input=decode_sequence)

# %% COMPILE
# opt = tfa.optimizers.AdamW(weight_decay=1e-4,
#                            learning_rate=1e-4,  # TODO: 1e-4 for transformer, and 1e-5 for backbone
#                            beta_1=0.9,
#                            beta_2=0.999,
#                            epsilon=1e-8,
#                            amsgrad=False
#                            )

opt = AdamWLearningRateMultiplier(weight_decay=1e-4,
                                  learning_rate=1e-4,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-8,
                                  amsgrad=False,
                                  lr_multipliers={"backbone": 0.1},
                                  )
detr.compile(optimizer=opt,
             loss=hungarian,
             )

# %% TRAIN
detr.fit(dataset,
         epochs=3,
         steps_per_epoch=50
         )
