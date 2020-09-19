import datetime
import inspect

import matplotlib.pyplot as plt
import tensorflow as tf

from chambers.utils.boxes import box_cxcywh_to_yxyx


def timestamp_now():
    dt = str(datetime.datetime.now())
    return "_".join(dt.split(" ")).split(".")[0]


def normalize_image(image):
    image = tf.cast(image, dtype=tf.float32)
    channel_avg = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    channel_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    image = (image / 255.0 - channel_avg) / channel_std
    return image


def denormalize_image(image):
    image = tf.cast(image, dtype=tf.float32)
    channel_avg = tf.constant([0.485, 0.456, 0.406])
    channel_std = tf.constant([0.229, 0.224, 0.225])

    image = image * channel_std
    image = image + channel_avg
    image = image * 255.0
    image = tf.cast(image, tf.uint8)
    return image


def plot_results(img, labels, probs, boxes):
    CLASSES = [
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

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    plt.figure(figsize=(16, 10))
    plt.imshow(img)
    ax = plt.gca()
    for cl, p, (xmin, ymin, xmax, ymax), c in zip(
            labels, probs, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{CLASSES[cl]}: {p:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()


def plot_img_boxes_cxcywh(img, boxes):
    boxes = box_cxcywh_to_yxyx(boxes)

    colors = [[1.0, 0., 0.]]
    box_img = tf.image.draw_bounding_boxes([img], [boxes], colors)
    box_img = tf.cast(box_img, tf.uint8)
    box_img = box_img[0]

    plt.imshow(box_img.numpy())
    plt.show()


def plot_img_boxes_yxyx(img, boxes):
    colors = [[1.0, 0., 0.]]
    box_img = tf.image.draw_bounding_boxes([img], [boxes], colors)
    box_img = tf.cast(box_img, tf.uint8)
    box_img = box_img[0]

    plt.imshow(box_img.numpy())
    plt.show()


def deserialize_object(identifier, module_objects, module_name, **kwargs):
    if type(identifier) is str:
        obj = module_objects.get(identifier)
        if obj is None:
            raise ValueError('Unknown ' + module_name + ':' + identifier)
        if inspect.isclass(obj):
            obj = obj(**kwargs)
        elif callable(obj):
            obj = obj
        return obj

    else:
        raise ValueError('Could not interpret serialized ' + module_name +
                         ': ' + identifier)
