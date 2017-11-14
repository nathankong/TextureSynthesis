import tensorflow as tf
import numpy as np

from VGGWeights import *
from Utils import *

HEIGHT = 256
WIDTH = 256
CHANNELS = 3

class Model:
    """ Takes in as argument the class that defines how to obtain pretrained weights.
        Tensorflow takes filters with shape:
            [filter_height, filter_width, in_channels, out_channels]

        Index: layer name (odd index is for biases) [filter shape]
        [out_channels, in_channels, height, width]
        0:  conv1_1 [64x3x3x3]
        2:  conv1_2 [64x64x3x3]
        4:  conv2_1 [128x64x3x3]
        6:  conv2_2 [128x128x3x3]
        8:  conv3_1 [256x128x3x3]
        10: conv3_2 [256x256x3x3]
        12: conv3_3 [256x256x3x3]
        14: conv3_4 [256x256x3x3]
        16: conv4_1 [512x256x3x3]
        18: conv4_2 [512x512x3x3]
        20: conv4_3 [512x512x3x3]
        22: conv4_4 [512x512x3x3]
        24: conv5_1 [512x512x3x3]
        26: conv5_2 [512x512x3x3]
        28: conv5_3 [512x512x3x3]
        30: conv5_4 [512x512x3x3]
    """

    def __init__(self, weights):
        self.weights = weights
        self.utils = Utils()
        self.model = dict()
        self.model_created = False

    def build_model(self):
        # Shape: [out_channels, in_channels, filter_height, filter_width]
        all_weights = self.weights.get_normalized_vgg_weights()

        # Input image size
        #self.model["input"] = tf.placeholder(tf.float32, shape=(1, HEIGHT, WIDTH, CHANNELS))
        self.model["input"] = tf.Variable(np.zeros((1, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)

        ind = 0
        self.model["conv1_1"] = self._conv2d_relu(self.model["input"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv1_2"] = self._conv2d_relu(self.model["conv1_1"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        self.model["pool1"] = self._avg_pool(self.model["conv1_2"])

        ind += 2
        self.model["conv2_1"] = self._conv2d_relu(self.model["pool1"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv2_2"] = self._conv2d_relu(self.model["conv2_1"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        self.model["pool2"] = self._avg_pool(self.model["conv2_2"])

        ind += 2
        self.model["conv3_1"] = self._conv2d_relu(self.model["pool2"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv3_2"] = self._conv2d_relu(self.model["conv3_1"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv3_3"] = self._conv2d_relu(self.model["conv3_2"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv3_4"] = self._conv2d_relu(self.model["conv3_3"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        self.model["pool3"] = self._avg_pool(self.model["conv3_4"])

        ind += 2
        self.model["conv4_1"] = self._conv2d_relu(self.model["pool3"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv4_2"] = self._conv2d_relu(self.model["conv4_1"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv4_3"] = self._conv2d_relu(self.model["conv4_2"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv4_4"] = self._conv2d_relu(self.model["conv4_3"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        self.model["pool4"] = self._avg_pool(self.model["conv4_4"])

        ind += 2
        self.model["conv5_1"] = self._conv2d_relu(self.model["pool4"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv5_2"] = self._conv2d_relu(self.model["conv5_1"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv5_3"] = self._conv2d_relu(self.model["conv5_2"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        ind += 2
        self.model["conv5_4"] = self._conv2d_relu(self.model["conv5_3"], self._transpose_weights(all_weights[ind]), all_weights[ind+1])
        self.model["pool5"] = self._avg_pool(self.model["conv5_4"])

        self.model_created = True

    def get_model(self):
        return self.model

    def model_initialized(self):
        return self.model_created

    def print_model(self):
        for key in self.model:
            print key + " shape: " + str(self.model[key].get_shape().as_list())

    def _transpose_weights(self, weights):
        # This assumes weights format: [out_channels, in_channels, H, W]
        # Needs to be modified depending on format of pretrained weights.
        return np.transpose(weights, [2,3,1,0])

    def _conv2d_relu(self, input_layer, weights, bias, strides=[1,1,1,1], padding="SAME"):
        bias = tf.constant(bias)
        weights = tf.constant(weights)
        return self.utils.conv2d_relu(input_layer, weights, bias, strides, padding)

    def _avg_pool(self, input_layer, kernel_size=[1,2,2,1], stride=[1,2,2,1], padding="SAME"):
        return self.utils.avg_pool(input_layer, kernel_size, stride, padding)

if __name__ == "__main__":
    vgg_weights = VGGWeights('vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()
    my_model.print_model()

