import tensorflow as tf

from chambers.augmentations import box_normalize_xyxy, random_resize_min
from chambers.utils.boxes import box_xywh_to_xyxy
from datasets import CocoDetection
from models import build_detr_resnet50
from utils import read_jpeg, normalize_image, denormalize_image, absolute2relative, plot_results, build_mask, remove_padding

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
dataset = dataset.map(lambda x, boxes, labels: (x, build_mask(x), boxes, labels))
dataset = dataset.padded_batch(batch_size=BATCH_SIZE,
                               # padded_shapes=((None, None, 3), (None, 4), (None,)),
                               padded_shapes=((None, None, 3), (None, None), (None, 4), (None,)),
                               padding_values=(
                                   tf.constant(-1.), tf.constant(True), tf.constant(-1.), tf.constant(-1)))

it = iter(dataset)

# %%
# x, boxes, labels = next(it)
x, mask, boxes, labels = next(it)
print("X SHAPE:", x.shape)
print("BOXES SHAPE:", boxes.shape)

# %%
detr = build_detr_resnet50()
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

# %%
xf = detr.backbone(x)
maskf = detr.downsample_masks(mask, xf)
xf.shape
maskf.shape

pos_encoding = detr.pos_encoder(maskf)
pos_encoding.shape

xfp = detr.input_proj(xf)
xfp.shape

hs = detr.transformer(xfp, maskf, detr.query_embed, pos_encoding)[0]
hs.shape

output_classes = detr.class_embed(hs)
output_classes.shape

bbox_embeding = detr.bbox_embed(hs)
bbox_embeding.shape

output_bbox = tf.sigmoid(bbox_embeding)

outputs = {'pred_logits': output_classes[-1],
          'pred_boxes': output_bbox[-1]}

# %%
outputs = detr((x, mask), post_process=False)

print("PRED LOGITS:", outputs["pred_logits"].shape)
print("PRED BOXES:", outputs["pred_boxes"].shape)

# %%
outputs = detr((x, mask), post_process=True)

# %% Show predictions
v_idx = 1
x_v = x[v_idx]
mask_v = mask[v_idx]
labels_v, scores, boxes_v = [outputs[k][v_idx].numpy() for k in ['labels', 'scores', 'boxes']]

keep = scores > 0.7
labels_v = labels_v[keep]
scores = scores[keep]
boxes_v = boxes_v[keep]

x_v = remove_padding(x_v, mask_v)
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
