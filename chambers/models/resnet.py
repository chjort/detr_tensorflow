import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, ReLU, MaxPool2D

from chambers.layers.batch_norm import FrozenBatchNorm2D


def ResNet50Backbone(input_shape, replace_stride_with_dilation=[False, False, False], name="resnet50"):
    inputs = tf.keras.layers.Input(input_shape)
    x = ConvBlock(inputs)
    x = ResidualBlock(x, num_bottlenecks=3, dim1=64, dim2=256, strides=1, replace_stride_with_dilation=False)
    x = ResidualBlock(x, num_bottlenecks=4, dim1=128, dim2=512, strides=2,
                      replace_stride_with_dilation=replace_stride_with_dilation[0])
    x = ResidualBlock(x, num_bottlenecks=6, dim1=256, dim2=1024, strides=2,
                      replace_stride_with_dilation=replace_stride_with_dilation[1])
    x = ResidualBlock(x, num_bottlenecks=3, dim1=512, dim2=2048, strides=2,
                      replace_stride_with_dilation=replace_stride_with_dilation[2])

    model = tf.keras.models.Model(inputs, x, name=name)
    return model


def ResNet101Backbone(input_shape, replace_stride_with_dilation=[False, False, False], name="resnet101"):
    inputs = tf.keras.layers.Input(input_shape)
    x = ConvBlock(inputs)
    x = ResidualBlock(x, num_bottlenecks=3, dim1=64, dim2=256, strides=1, replace_stride_with_dilation=False)
    x = ResidualBlock(x, num_bottlenecks=4, dim1=128, dim2=512, strides=2,
                      replace_stride_with_dilation=replace_stride_with_dilation[0])
    x = ResidualBlock(x, num_bottlenecks=23, dim1=256, dim2=1024, strides=2,
                      replace_stride_with_dilation=replace_stride_with_dilation[1])
    x = ResidualBlock(x, num_bottlenecks=3, dim1=512, dim2=2048, strides=2,
                      replace_stride_with_dilation=replace_stride_with_dilation[2])

    model = tf.keras.models.Model(inputs, x, name=name)
    return model


def ConvBlock(x):
    x = ZeroPadding2D(3)(x)
    x = Conv2D(64, kernel_size=7, strides=2, padding='valid',
               use_bias=False)(x)
    x = FrozenBatchNorm2D()(x)
    x = ReLU()(x)
    x = ZeroPadding2D(1)(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='valid')(x)
    return x


def ResidualBlock(x, num_bottlenecks, dim1, dim2, strides=1, replace_stride_with_dilation=False):
    if replace_stride_with_dilation:
        strides = 1
        dilation = 2
    else:
        dilation = 1

    x = BottleNeck(x, dim1, dim2, strides=strides, downsample=True)

    for i in range(1, num_bottlenecks):
        x = BottleNeck(x, dim1, dim2, dilation=dilation)

    return x


def BottleNeck(x, dim1, dim2, strides=1, dilation=1, downsample=False):
    identity = x

    x = Conv2D(dim1, kernel_size=1, use_bias=False)(x)
    x = FrozenBatchNorm2D()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(dilation)(x)
    x = Conv2D(dim1, kernel_size=3, strides=strides, dilation_rate=dilation,
               use_bias=False)(x)
    x = FrozenBatchNorm2D()(x)
    x = ReLU()(x)

    x = Conv2D(dim2, kernel_size=1, use_bias=False)(x)
    x = FrozenBatchNorm2D()(x)

    if downsample:
        identity = Conv2D(dim2, kernel_size=1, strides=strides, use_bias=False)(identity)
        identity = FrozenBatchNorm2D()(identity)

    x = x + identity
    x = ReLU()(x)

    return x
