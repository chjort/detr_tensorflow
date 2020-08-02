import matplotlib.pyplot as plt
import tensorflow as tf


def read_jpeg(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def resize(image, min_side=800, max_side=1333):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cur_min_side = tf.minimum(w, h)
    cur_max_side = tf.maximum(w, h)

    min_side = tf.cast(min_side, tf.float32)
    max_side = tf.cast(max_side, tf.float32)
    scale = tf.minimum(max_side / cur_max_side,
                       min_side / cur_min_side)
    nh = tf.cast(scale * h, tf.int32)
    nw = tf.cast(scale * w, tf.int32)

    image = tf.image.resize(image, (nh, nw))
    return image


def build_mask(image):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    return tf.zeros((h, w), dtype=tf.bool)


def cxcywh2xyxy(boxes):
    cx, cy, w, h = [boxes[..., i] for i in range(4)]

    xmin, ymin = cx - w * 0.5, cy - h * 0.5
    xmax, ymax = cx + w * 0.5, cy + h * 0.5

    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return boxes


def absolute2relative(boxes, img_size):
    width, height = img_size
    scale = tf.constant([width, height, width, height], dtype=tf.float32)
    boxes *= scale
    return boxes


def xyxy2xywh(boxes):
    xmin, ymin, xmax, ymax = [boxes[..., i] for i in range(4)]
    return tf.stack([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)


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


def preprocess_image(image):
    image = resize(image, min_side=800, max_side=1333)
    image = normalize_image(image)

    return image, build_mask(image)


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
