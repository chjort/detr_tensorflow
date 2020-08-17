import tensorflow as tf
import tensorflow_datasets as tfds

from chambers.augmentations import random_resize_min, box_normalize_cxcywh, box_denormalize_yxyx
from chambers.utils.boxes import box_yxyx_to_cxcywh, box_xywh_to_cxcywh
from datasets import CocoDetection
from utils import normalize_image, read_jpeg


def load_coco(coco_path, split, batch_size):
    def augment(img, boxes, labels):
        # TODO: Random Horizontal Flip

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
        boxes = box_xywh_to_cxcywh(boxes)
        boxes = box_normalize_cxcywh(boxes, img)
        return img, boxes, labels

    coco_data = CocoDetection(coco_path, partition=split + "2017")  # boxes [x0, y0, w, h]
    dataset = tf.data.Dataset.from_generator(lambda: coco_data, output_types=(tf.string, tf.float32, tf.int32))
    # dataset = dataset.repeat()
    dataset = dataset.map(lambda img_path, boxes, labels: (read_jpeg(img_path), boxes, tf.cast(labels, tf.float32)))
    dataset = dataset.map(augment)
    dataset = dataset.map(normalize)
    dataset = dataset.map(lambda img, boxes, labels: (img, tf.concat([boxes, tf.expand_dims(labels, 1)], axis=1)))

    dataset = dataset.padded_batch(batch_size=batch_size,
                                   # padded_shapes=((None, None, 3), (None, 4), (None,)),
                                   # padding_values=(tf.constant(-1.), tf.constant(-1.), tf.constant(-1.))
                                   padded_shapes=((None, None, 3), (None, 5)),
                                   padding_values=(tf.constant(-1.), tf.constant(-1.))
                                   )

    N = len(coco_data)
    return dataset, N


def load_coco_tf(split, batch_size):
    dataset, info = tfds.load("coco/2017",
                              split=split,
                              data_dir="/datadrive/crr/tensorflow_datasets",
                              with_info=True)

    print(info.features)  # bbox format: [y_min, x_min, y_max, x_max]

    def augment(img, boxes, labels):
        # TODO: Random Horizontal Flip

        min_sides = [800]
        # min_sides = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

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
        boxes = box_normalize_cxcywh(boxes, img)
        return img, boxes, labels

    dataset = dataset.filter(lambda x: tf.shape(x["objects"]["label"])[0] > 0)  # remove elements with no annotations
    dataset = dataset.map(lambda x: (x["image"], x["objects"]["bbox"], x["objects"]["label"]))
    dataset = dataset.map(
        lambda x, boxes, labels: (x, box_yxyx_to_cxcywh(box_denormalize_yxyx(boxes, x)), tf.cast(labels, tf.float32)))
    dataset = dataset.map(augment)
    dataset = dataset.map(normalize)
    dataset = dataset.map(lambda img, boxes, labels: (img, tf.concat([boxes, tf.expand_dims(labels, 1)], axis=1)))
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