import tensorflow as tf
import tensorflow_datasets as tfds

from chambers.augmentations import random_resize_min, box_denormalize_yxyx, \
    random_size_crop, resize, box_normalize_yxyx, random_flip_left_right
from chambers.utils.boxes import box_xywh_to_yxyx
from chambers.utils.image import read_jpeg, resnet_imagenet_normalize
from data.coco import CocoDetection

N_PARALLEL = -1

CLASSES_COCO = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
CLASSES_TF = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


def augment(img, boxes, labels):
    img, boxes = random_flip_left_right(img, boxes)

    min_sides = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    def _fn1(img, boxes, labels):
        img, boxes = random_resize_min(img, boxes, min_sides=min_sides, max_side=1333)
        return img, boxes, labels

    def _fn2(img, boxes, labels):
        img, boxes = random_resize_min(img, boxes, min_sides=[400, 500, 600], max_side=None)
        img, boxes, labels = random_size_crop(img, boxes, labels, min_size=384, max_size=600)
        img, boxes = random_resize_min(img, boxes, min_sides=min_sides, max_side=1333)
        return img, boxes, labels

    img, boxes, labels = tf.cond(tf.random.uniform([1], 0, 1) > 0.5,
                                 true_fn=lambda: _fn1(img, boxes, labels),
                                 false_fn=lambda: _fn2(img, boxes, labels)
                                 )

    return img, boxes, labels


def augment_val(img, boxes, labels):
    img, boxes = resize(img, boxes, min_side=800, max_side=1333)
    # img, boxes = random_flip_left_right(img, boxes)
    # img, boxes = resize(img, boxes, shape=(768, 768))

    return img, boxes, labels


def normalize(img, boxes, labels):
    img = resnet_imagenet_normalize(img)
    boxes = box_normalize_yxyx(boxes, img)
    return img, boxes, labels


def load_coco(coco_path, split, batch_size):
    coco_data = CocoDetection(coco_path, partition=split + "2017")  # boxes [x0, y0, w, h]
    dataset = tf.data.Dataset.from_generator(lambda: coco_data, output_types=(tf.string, tf.float32, tf.int32))
    dataset = dataset.filter(
        lambda img_path, boxes, labels: tf.shape(boxes)[0] > 0)  # remove elements with no annotations
    dataset = dataset.map(
        lambda img_path, boxes, labels: (read_jpeg(img_path), box_xywh_to_yxyx(boxes), tf.cast(labels, tf.float32)),
        num_parallel_calls=N_PARALLEL
    )
    dataset = dataset.cache()
    if split == "train":
        dataset = dataset.repeat()
        # dataset = dataset.shuffle(1024)
        dataset = dataset.map(augment, num_parallel_calls=N_PARALLEL)
    else:
        dataset = dataset.map(augment_val, num_parallel_calls=N_PARALLEL)
    dataset = dataset.filter(
        lambda img_path, boxes, labels: tf.shape(boxes)[0] > 0)  # remove elements with no annotations
    dataset = dataset.map(normalize, num_parallel_calls=N_PARALLEL)
    dataset = dataset.map(lambda img, boxes, labels: (img, tf.concat([boxes, tf.expand_dims(labels, 1)], axis=1)),
                          num_parallel_calls=N_PARALLEL)

    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=((None, None, 3), (None, 5)),
                                   padding_values=(tf.constant(-1.), tf.constant(-1.))
                                   )

    if split == "train":
        n_samples_no_label = 1021
    elif split == "val":
        n_samples_no_label = 43
    else:
        n_samples_no_label = 0
    n = len(coco_data) - n_samples_no_label
    return dataset, n


def load_coco_tf(split, batch_size, data_dir=None):
    dataset, info = tfds.load("coco/2017",
                              split=split,
                              data_dir=data_dir,
                              with_info=True)

    print(info.features)  # bbox format: [y_min, x_min, y_max, x_max]

    dataset = dataset.filter(lambda x: tf.shape(x["objects"]["label"])[0] > 0)  # remove elements with no annotations
    dataset = dataset.map(lambda x: (x["image"], x["objects"]["bbox"], x["objects"]["label"]),
                          num_parallel_calls=N_PARALLEL)
    dataset = dataset.map(
        lambda x, boxes, labels: (x, box_denormalize_yxyx(boxes, x), tf.cast(labels, tf.float32)),
        num_parallel_calls=N_PARALLEL)
    dataset = dataset.cache()
    if split == "train":
        dataset = dataset.repeat()
        # dataset = dataset.shuffle(1024)
        dataset = dataset.map(augment, num_parallel_calls=N_PARALLEL)
        # dataset = dataset.map(augment_val, num_parallel_calls=N_PARALLEL)
    else:
        dataset = dataset.map(augment_val, num_parallel_calls=N_PARALLEL)
    dataset = dataset.map(normalize, num_parallel_calls=N_PARALLEL)
    dataset = dataset.map(lambda img, boxes, labels: (img, tf.concat([boxes, tf.expand_dims(labels, 1)], axis=1)),
                          num_parallel_calls=N_PARALLEL)
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=((None, None, 3), (None, 5)),
                                   padding_values=(tf.constant(-1.), tf.constant(-1.))
                                   )

    if split == "train":
        n_samples_no_label = 1021
    elif split == "validation":
        n_samples_no_label = 48
    else:
        n_samples_no_label = 0
    n = info.splits[split].num_examples - n_samples_no_label
    return dataset, n
