import tensorflow as tf
import tensorflow_datasets as tfds

from chambers.augmentations import random_resize_min, box_normalize_cxcywh, box_denormalize_yxyx, flip_left_right, \
    random_size_crop, resize, box_normalize_yxyx
from chambers.utils.boxes import box_yxyx_to_cxcywh, box_xywh_to_yxyx
from datasets import CocoDetection
from utils import normalize_image, read_jpeg

N_PARALLEL = -1

def augment(img, boxes, labels):
    img, boxes = tf.cond(tf.random.uniform([1], 0, 1) > 0.5,
                         true_fn=lambda: flip_left_right(img, boxes),
                         false_fn=lambda: (img, boxes)
                         )

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

    return img, boxes, labels


def normalize(img, boxes, labels):
    img = normalize_image(img)
    boxes = box_yxyx_to_cxcywh(boxes)
    boxes = box_normalize_cxcywh(boxes, img)
    # boxes = box_normalize_yxyx(boxes, img)
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
        # dataset = dataset.map(augment, num_parallel_calls=N_PARALLEL)
        dataset = dataset.map(augment_val, num_parallel_calls=N_PARALLEL)
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
    N = len(coco_data) - n_samples_no_label
    return dataset, N


def load_coco_tf(split, batch_size):
    dataset, info = tfds.load("coco/2017",
                              split=split,
                              data_dir="/datadrive/crr/tensorflow_datasets",
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
        # dataset = dataset.map(augment, num_parallel_calls=N_PARALLEL)
        dataset = dataset.map(augment_val, num_parallel_calls=N_PARALLEL)
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
    N = info.splits[split].num_examples - n_samples_no_label
    return dataset, N
