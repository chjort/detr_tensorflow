import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, ReLU, MaxPool2D, BatchNormalization


def ResNet50Backbone(input_shape, replace_stride_with_dilation=[False, False, False], name="resnet50"):
    with tf.name_scope(name):
        inputs = tf.keras.layers.Input(input_shape)
        x = ConvBlock(inputs, name_prefix=name + "/")
        x = ResidualBlock(x, num_bottlenecks=3, dim1=64, dim2=256, strides=1, replace_stride_with_dilation=False,
                          name_prefix=name + "/layer1/")
        x = ResidualBlock(x, num_bottlenecks=4, dim1=128, dim2=512, strides=2,
                          replace_stride_with_dilation=replace_stride_with_dilation[0],
                          name_prefix=name + "/layer2/")
        x = ResidualBlock(x, num_bottlenecks=6, dim1=256, dim2=1024, strides=2,
                          replace_stride_with_dilation=replace_stride_with_dilation[1],
                          name_prefix=name + "/layer3/")
        x = ResidualBlock(x, num_bottlenecks=3, dim1=512, dim2=2048, strides=2,
                          replace_stride_with_dilation=replace_stride_with_dilation[2],
                          name_prefix=name + "/layer4/")

    model = tf.keras.models.Model(inputs, x, name=name)
    return model


def ResNet101Backbone(input_shape, replace_stride_with_dilation=[False, False, False], name="resnet101"):
    inputs = tf.keras.layers.Input(input_shape)
    x = ConvBlock(inputs, name_prefix=name + "/")
    x = ResidualBlock(x, num_bottlenecks=3, dim1=64, dim2=256, strides=1, replace_stride_with_dilation=False,
                      name_prefix=name + "/layer1/")
    x = ResidualBlock(x, num_bottlenecks=4, dim1=128, dim2=512, strides=2,
                      replace_stride_with_dilation=replace_stride_with_dilation[0],
                      name_prefix=name + "/layer2/")
    x = ResidualBlock(x, num_bottlenecks=23, dim1=256, dim2=1024, strides=2,
                      replace_stride_with_dilation=replace_stride_with_dilation[1],
                      name_prefix=name + "/layer3/")
    x = ResidualBlock(x, num_bottlenecks=3, dim1=512, dim2=2048, strides=2,
                      replace_stride_with_dilation=replace_stride_with_dilation[2],
                      name_prefix=name + "/layer4/")

    model = tf.keras.models.Model(inputs, x, name=name)
    return model


def ConvBlock(x, name_prefix=""):
    x = ZeroPadding2D(3, name=name_prefix + "pad1")(x)
    x = Conv2D(64, kernel_size=7, strides=2, padding='valid',
               use_bias=False, name=name_prefix + "conv1")(x)
    x = BatchNormalization(name=name_prefix + "bn1")(x)
    x = ReLU(name=name_prefix + "relu")(x)
    x = ZeroPadding2D(1, name=name_prefix + "pad2")(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='valid')(x)
    return x


def ResidualBlock(x, num_bottlenecks, dim1, dim2, strides=1, replace_stride_with_dilation=False, name_prefix=""):
    with tf.name_scope(name_prefix):
        if replace_stride_with_dilation:
            strides = 1
            dilation = 2
        else:
            dilation = 1

        x = BottleNeck(x, dim1, dim2, strides=strides, downsample=True, name_prefix=name_prefix + "0/")

        for i in range(1, num_bottlenecks):
            x = BottleNeck(x, dim1, dim2, dilation=dilation, name_prefix=name_prefix + str(i) + "/")

    return x


def BottleNeck(x, dim1, dim2, strides=1, dilation=1, downsample=False, name_prefix=""):
    with tf.name_scope(name_prefix):
        identity = x

        x = Conv2D(dim1, kernel_size=1, use_bias=False, name=name_prefix + "conv1")(x)
        x = BatchNormalization(name=name_prefix + "bn1")(x)
        relu = ReLU(name=name_prefix + "relu")
        x = relu(x)

        x = ZeroPadding2D(dilation)(x)
        x = Conv2D(dim1, kernel_size=3, strides=strides, dilation_rate=dilation,
                   use_bias=False, name=name_prefix + "conv2")(x)
        x = BatchNormalization(name=name_prefix + "bn2")(x)
        x = relu(x)

        x = Conv2D(dim2, kernel_size=1, use_bias=False, name=name_prefix + "conv3")(x)
        x = BatchNormalization(name=name_prefix + "bn3")(x)

        if downsample:
            downsample_layer = tf.keras.Sequential([
                Conv2D(dim2, kernel_size=1, strides=strides, use_bias=False, name="0"),
                BatchNormalization(name="1")
            ], name=name_prefix + "downsample")
            identity = downsample_layer(identity)

        x = x + identity
        x = relu(x)

    return x
