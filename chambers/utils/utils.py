import io
import datetime
import inspect

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from chambers.augmentations import box_denormalize_yxyx
from chambers.utils.boxes import box_cxcywh_to_yxyx


def timestamp_now():
    dt = str(datetime.datetime.now())
    return "_".join(dt.split(" ")).split(".")[0]


def imshow(img):
    img = np.array(img)
    plt.imshow(img)
    plt.show()


def plot_results(img, boxes, labels=None, probs=None, colors=None, linewidth=3, text_color="yellow", text_alpha=0.5,
                 fontsize=None, figsize=None, return_img=False):
    """
    Plots the bounding boxes and labels onto an image.

    :param img: RGB image with shape [h, w, c] with pixel values between 0 and 255.
    :param boxes: Bounding boxes with shape [n, 4] and format [y0, x0, y1, x1]. Boxes should be normalized to be between
        0 and 1.
    :param labels: List of labels
    :param probs: List of label probabilities
    :param colors: List of colors to cycle through
    :return:
    """

    boxes = box_denormalize_yxyx(boxes, img)

    img = np.array(img)
    boxes = np.array(boxes)
    if labels is not None:
        labels = np.array(labels)
    if probs is not None:
        probs = np.array(probs)
    if colors is None:
        colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    if figsize is None:
        figsize = (img.shape[1]/100, img.shape[0]/100)

    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    ax = plt.gca()

    for i in range(len(boxes)):
        y0, x0, y1, x1 = boxes[i]
        color = colors[i % len(colors)]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color, linewidth=linewidth))

        text = None
        if labels is not None:
            label = labels[i]
            text = f'{label}'
        if probs is not None:
            p = probs[i]
            text = text + f': {p:0.2f}'

        if text is not None:
            if fontsize is None:
                fontsize_ratio = 1.8e-05
                fontsize = (figsize[0] * figsize[1]) * 100 * fontsize_ratio
                fontsize = np.round(fontsize, 0).astype(int)
            ax.text(x0, y0+fontsize, text, fontsize=fontsize, bbox=dict(facecolor=text_color, alpha=text_alpha))

    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if return_img:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        out_img = tf.image.decode_png(buf.getvalue(), channels=4)
        return out_img
    else:
        plt.show()


def plot_img_boxes_cxcywh(img, boxes, colors=None):
    boxes = box_cxcywh_to_yxyx(boxes)
    plot_img_boxes_yxyx(img, boxes, colors)


def plot_img_boxes_yxyx(img, boxes, colors=None):
    if colors is None:
        colors = [[1.0, 0.0, 0.0]]
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
