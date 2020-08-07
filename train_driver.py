import tensorflow as tf

from chambers.augmentations import box_normalize_xyxy, random_resize_min
from chambers.utils.boxes import box_xywh_to_xyxy, absolute2relative
from chambers.utils.masking import remove_padding_image
from datasets import CocoDetection
from models import build_detr_resnet50
from utils import read_jpeg, normalize_image, denormalize_image, plot_results

COCO_PATH = "/home/crr/datasets/coco"
# COCO_PATH = "/home/ch/datasets/coco"
BATCH_SIZE = 2


# %%
def augment(img, boxes, labels):
    # TODO: Random Horizontal Flip

    # img, boxes = resize(img, boxes, 800.0, 1333.0)

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


coco_data = CocoDetection(COCO_PATH, partition='val2017')  # boxes [x0, y0, w, h]
dataset = tf.data.Dataset.from_generator(lambda: coco_data, (tf.string, tf.float32, tf.int32))
dataset = dataset.map(lambda img_path, boxes, labels: (read_jpeg(img_path), box_xywh_to_xyxy(boxes), labels))
dataset = dataset.map(augment)
dataset = dataset.map(normalize)
dataset = dataset.padded_batch(batch_size=BATCH_SIZE,
                               padded_shapes=((None, None, 3), (None, 4), (None,)),
                               padding_values=(tf.constant(-1.), tf.constant(-1.), tf.constant(-1)))

it = iter(dataset)

# %%
x, boxes, labels = next(it)
print("X SHAPE:", x.shape)
print("BOXES SHAPE:", boxes.shape)
print("LABELS SHAPE:", labels.shape)

# %%
detr = build_detr_resnet50(mask_value=-1.)
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

# %%
class_pred, box_pred = detr(x)
print("PRED LOGITS:", class_pred.shape)
print("PRED BOXES:", box_pred.shape)

# %%
scores, boxes_v, labels_v = detr.post_process(class_pred, box_pred)

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
