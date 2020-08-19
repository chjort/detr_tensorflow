import tensorflow as tf

from chambers.losses import HungarianLoss
from chambers.utils.boxes import absolute2relative
from chambers.utils.masking import remove_padding_image
from models import build_detr_resnet50
from tf_datasets import load_coco_tf, load_coco
from utils import denormalize_image, plot_results

# %%
BATCH_SIZE = 2
# dataset, N = load_coco_tf("train", BATCH_SIZE)
# dataset, N = load_coco("/datadrive/crr/datasets/coco", "train", BATCH_SIZE)
dataset, N = load_coco("/home/ch/datasets/coco", "train", BATCH_SIZE)

# %%
decode_sequence = False
detr = build_detr_resnet50(num_classes=91,
                           num_queries=100,
                           mask_value=-1.,
                           return_decode_sequence=decode_sequence)
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

hungarian = HungarianLoss(mask_value=-1., sequence_input=decode_sequence)

# %% COMPILE
detr.compile("adam",
             loss=hungarian,
             )

# %% TRAIN
# detr + loss_placeholder (no sequence) + 1 bsz + 800 min: 773, 707, 613 ms per step
# detr + loss_placeholder (no sequence) + 2 bsz + 800 min: 1000+, 978, 976 ms per step

# detr + hungarian (no sequence) + 1 bsz + 800 min: 778, 711, 618 ms per step

detr.fit(dataset,
         epochs=3,
         steps_per_epoch=50
         )

# %%
it = iter(dataset)

# %%
x, y = next(it)
print("X SHAPE:", x.shape)
print("Y SHAPE:", y.shape)

# n_ypad = tf.reduce_sum(tf.cast(tf.equal(x[:, :, 0, 0], -1.), tf.float32), axis=1)
# n_xpad = tf.reduce_sum(tf.cast(tf.equal(x[:, 0, :, 0], -1.), tf.float32), axis=1)

#%%
from chambers.augmentations import random_size_crop

xr, yr = random_size_crop(x[0], y[0][:, :4], min_size=128, max_size=500)

# %%
y_pred = detr.predict(x)
print("PRED:", y_pred.shape)

# %%
loss = hungarian(y, y_pred)
print(loss)

# %%
scores, boxes_v, labels_v = detr.post_process(y_pred)

# NOTE: The class indices of the tf.dataset does not match the original coco class indices, which the pretrained DETR
#   model were trained on. This causes a high loss, even though the prediction is correct.
print(labels_v, y[:, :, -1])

# %% Show predictions
v_idx = 0
x_v = x[v_idx]
labels_v = labels_v[v_idx].numpy()
scores = scores[v_idx].numpy()
boxes_v = boxes_v[v_idx].numpy()

keep = scores > 0.7
labels_v = labels_v[keep]
scores = scores[keep]
boxes_v = boxes_v[keep]

x_v = remove_padding_image(x_v, -1)
boxes_v = absolute2relative(boxes_v, (x_v.shape[1], x_v.shape[0])).numpy()

x_v = denormalize_image(x_v)
plot_results(x_v.numpy(), labels_v, scores, boxes_v)

# %% Show ground truth
from chambers.utils.boxes import box_cxcywh_to_yxyx
import matplotlib.pyplot as plt

i = 0

x_v = denormalize_image(x)
boxes = y[..., :4]
boxes_viz = box_cxcywh_to_yxyx(boxes)
# boxes_viz = boxes
x_v = x_v[i]
boxes_viz = boxes_viz[i]

colors = [[1.0, 0., 0.]]
box_img = tf.image.draw_bounding_boxes([x_v], [boxes_viz], colors)
box_img = tf.cast(box_img, tf.uint8)
box_img = box_img[0]

plt.imshow(box_img.numpy())
plt.show()
