import tensorflow as tf

class Utils:
    """ Defines a set of utility functions to build layers. Tensorflow
        filter format is: [H, W, in_channels, out_channels]
    """

    def conv2d_relu(self, layer, filters, bias, strides, padding):
        return tf.nn.relu(tf.nn.conv2d(layer, filter=filters, strides=strides, padding=padding) + bias)

    def avg_pool(self, layer, kernel_size, stride, padding):
        return tf.nn.avg_pool(layer, ksize=kernel_size, strides=stride, padding=padding)

