from chambers.utils.boxes import absolute2relative
from chambers.utils.data import time_dataset
from chambers.utils.masking import remove_padding_image
from models import build_detr_resnet50
from tf_datasets import load_coco
from utils import denormalize_image, plot_results, plot_img_boxes_cxcywh

# %%
BATCH_SIZE = 2
# dataset, N = load_coco_tf("train", BATCH_SIZE)
dataset, N = load_coco("/datadrive/crr/datasets/coco", "train", BATCH_SIZE)
# dataset, N = load_coco("/home/ch/datasets/coco", "train", BATCH_SIZE)

# %%
# train cache 1: 1976.1714327335358
# train cache 2: 1398.1613807678223
#
# val: 32.44592475891113
# val augment: 101.61271572113037
# val cache 1: 38.116862773895264
# val cache 2: 18.88991403579712
# val cache 2 aug: 85.98375701904297

time_dataset(dataset, n=N)

# %%
it = iter(dataset)

# %%
x, y = next(it)
print("X SHAPE:", x.shape)
print("Y SHAPE:", y.shape)

# %%
i = 1
img = denormalize_image(x[i])
boxes = y[..., :4][i]
plot_img_boxes_cxcywh(img, boxes)

# %%
decode_sequence = False
detr = build_detr_resnet50(num_classes=91,
                           num_queries=100,
                           mask_value=-1.,
                           return_decode_sequence=decode_sequence)
detr.build()
detr.load_from_pickle('checkpoints/detr-r50-e632da11.pickle')

# %%
y_pred = detr.predict(x)
print("PRED:", y_pred.shape)

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