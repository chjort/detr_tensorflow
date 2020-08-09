import tensorflow as tf

from chambers.augmentations import box_normalize_xyxy, random_resize_min
from chambers.losses import HungarianLoss
from chambers.utils.boxes import box_xywh_to_xyxy, absolute2relative
from chambers.utils.masking import remove_padding_image
from datasets import CocoDetection
from models import build_detr_resnet50
from utils import read_jpeg, normalize_image, denormalize_image, plot_results


def loss_placeholder(y_true, y_pred):
    y_true_labels = y_true[..., -1]  # [1]
    y_pred_logits = y_pred[..., 4:]  # [1]
    y_pred = y_pred_logits[:, :1, :]
    y_true = tf.one_hot(tf.cast(y_true_labels, tf.int32), depth=92)[:, :1, :]
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)


COCO_PATH = "/home/crr/datasets/coco"
# COCO_PATH = "/home/ch/datasets/coco"
BATCH_SIZE = 2


# %%
def augment(img, boxes, labels):
    # TODO: Random Horizontal Flip

    # min_sides = [544]
    min_sides = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # TODO: Random choice (50%)
    img, boxes = random_resize_min(img, boxes, min_sides=min_sides, max_side=1333)
    # ,
    # img, boxes = random_resize_min(img, boxes, min_sides=[400, 500, 600], max_side=None)
    # img, boxes = random_size_crop(img, boxes, min_size=384, max_size=600)
    # img, boxes = random_resize_min(img, boxes, min_sides=min_sides, max_side=1333)
    # TODO: End random choice

    return img, boxes, labels


def normalize(img, boxes, labels):
    img = normalize_image(img)
    boxes = box_normalize_xyxy(boxes, img)
    return img, boxes, labels


coco_data = CocoDetection(COCO_PATH, partition='train2017')  # boxes [x0, y0, w, h]
dataset = tf.data.Dataset.from_generator(lambda: coco_data, output_types=(tf.string, tf.float32, tf.int32))
# dataset = dataset.repeat()
dataset = dataset.map(lambda img_path, boxes, labels: (read_jpeg(img_path), box_xywh_to_xyxy(boxes), tf.cast(labels, tf.float32)))
dataset = dataset.map(augment)
dataset = dataset.map(normalize)
# dataset = dataset.map(lambda img, boxes, labels: (img, tf.concat([boxes, labels], axis=0)))
dataset = dataset.map(lambda img, boxes, labels: (img, tf.concat([boxes, tf.expand_dims(labels, 1)], axis=1)))

dataset = dataset.padded_batch(batch_size=BATCH_SIZE,
                               # padded_shapes=((None, None, 3), (None, 4), (None,)),
                               # padding_values=(tf.constant(-1.), tf.constant(-1.), tf.constant(-1.))
                               padded_shapes=((None, None, 3), (None, 5)),
                               padding_values=(tf.constant(-1.), tf.constant(-1.))
                               )

N = len(coco_data)

#%%
for i, (x, y) in dataset.take(22).enumerate():
    print(i, x.shape, y.shape)
    # TODO: Fix sample 21. It breaks the 'padded_batch' for some reason.



# %%
decode_sequence = False
detr = build_detr_resnet50(mask_value=-1., return_decode_sequence=decode_sequence)
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

hungarian = HungarianLoss(mask_value=-1., sequence_input=decode_sequence)

# %% COMPILE
detr.compile("adam",
             loss=hungarian,
             # loss=loss_placeholder
             )

# %% TRAIN
# detr + loss_placeholder: 1s per step
# detr + hungarian: 1s per step

detr.fit(dataset,
         epochs=1,
         # steps_per_epoch=N
         )

# %%
it = iter(dataset)

# import time
# st = time.time()
# for x, y in dataset.take(10):
#     y_pred = detr(x)
#     loss = hungarian(y, y_pred)
#     print(loss)
# print(time.time() - st)

# eager: 59.922558307647705
# graph: 64.54768967628479 (excessive retracing)
# graph: 61.242064237594604

# %%
# x, y_true_boxes, y_true_logits = next(it)
x, y = next(it)
print("X SHAPE:", x.shape)
print("BOXES SHAPE:", y.shape)
# print("LABELS SHAPE:", y_true_logits.shape)

# %%
# y_pred_logits, y_pred_boxes = detr(x)
y_pred = detr(x)
# print("PRED LOGITS:", y_pred_logits.shape)
print("PRED BOXES:", y_pred.shape)

# %%
# loss = hungarian((y_true_logits, y_true_boxes), (y_pred_logits, y_pred_boxes))
loss = hungarian(y, y_pred)
print(loss)

# %%
scores, boxes_v, labels_v = detr.post_process(y_pred)

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
# boxes_viz = boxes
# boxes_viz = box_xyxy_to_yxyx(boxes_viz)
#
# colors = [[1.0, 0., 0.]]
# box_img = tf.image.draw_bounding_boxes([x_v], [boxes_viz], colors)
# box_img = tf.cast(box_img, tf.uint8)
# box_img = box_img[0]
#
# plt.imshow(box_img.numpy())
# plt.show()
