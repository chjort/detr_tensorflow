import tensorflow as tf
from chambers.losses import HungarianLoss, pairwise_softmax, pairwise_l1, pairwise_giou
from chambers.utils.boxes import absolute2relative
from chambers.utils.masking import remove_padding_image
from models import build_detr_resnet50
from tf_datasets import load_coco
from utils import denormalize_image, plot_results, plot_img_boxes_cxcywh, plot_img_boxes_yxyx

# %%
BATCH_SIZE = 1
# dataset, N = load_coco_tf("train", BATCH_SIZE)
# dataset, N = load_coco("/datadrive/crr/datasets/coco", "train", BATCH_SIZE)
dataset, N = load_coco("/home/ch/datasets/coco", "val", BATCH_SIZE)

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

# %% Show ground truth
i = 0
img = denormalize_image(x[0])
boxes = y[..., :4][0]

plot_img_boxes_cxcywh(img, boxes)

# %%
from chambers.augmentations import random_size_crop_ch, random_size_crop
from chambers.utils.boxes import box_cxcywh_to_yxyx

x_ = x[0]
y_ = box_cxcywh_to_yxyx(y[0, ..., :4])
l_ = y[0, ..., -1]

# %%
min_size = 128
max_size = 500

hw = tf.random.uniform([2], min_size, max_size + 1, dtype=tf.int32)
h = hw[0]
w = hw[1]

input_shape = tf.shape(x_)
ylim = input_shape[0] - h
xlim = input_shape[1] - w
y0 = tf.random.uniform([], 0, ylim, dtype=tf.int32)
x0 = tf.random.uniform([], 0, xlim, dtype=tf.int32)

img_h = input_shape[0]
img_w = input_shape[1]
y0_n = y0 / img_h
x0_n = x0 / img_w
h_n = h / img_h
w_n = w / img_w

# TODO: Crop such that there is always a bounding box in the crop


# %%
min_size = 128
max_size = 500

hw = tf.random.uniform([2], min_size, max_size + 1, dtype=tf.int32)
h = hw[0]
w = hw[1]

min_dim = tf.minimum(h, w)
max_dim = tf.maximum(h, w)
aspect_ratio = [min_dim / max_dim, max_dim / min_dim]

img_shape = tf.shape(x_)
img_h = img_shape[0]
img_w = img_shape[0]
img_area = img_h * img_w
crop_area = h * w
area_ratio = crop_area / img_area

y_b = tf.expand_dims(y_, 0)
begin, size, bboxes = tf.image.sample_distorted_bounding_box(img_shape, [y_],
                                                             min_object_covered=0.1,
                                                             aspect_ratio_range=aspect_ratio,
                                                             area_range=[0.1, 0.105]
                                                             )

begin, size, bboxes = random_size_crop(x_, y_, l_, 128, 500)
plot_img_boxes_yxyx(denormalize_image(x_), bboxes[0])

# %%
xr, yr = random_size_crop_ch(x_, y_, min_size=128, max_size=500)
xr.shape
plot_img_boxes_yxyx(denormalize_image(xr), yr)

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
