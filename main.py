import numpy as np
import tensorflow as tf
import os
import sys

import TextureSynthesis as ts
from VGGWeights import *
from model import *

def main():
    # Load VGG-19 weights and build model
    vgg_weights = VGGWeights('vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()

    # Load tensorflow session
    sess = tf.Session()

    #layer_weights = {"conv1_1": 1e9, "conv2_1": 1e9, "conv3_1": 1e9, "conv4_1": 1e9, "conv5_1": 1e9}
    #layer_weights = {"conv1_1": 1e9, "conv2_1": 1e9, "conv3_1": 1e9, "conv4_1": 1e9}
    #layer_weights = {"conv1_1": 1e9, "conv2_1": 1e9, "conv3_1": 1e9}
    #layer_weights = {"conv1_1": 1e9, "conv2_1": 1e9}
    #layer_weights = {"conv1_1": 1e9}

    layer_weights = {"conv1_1": 1e9, "pool1": 1e9, "pool2": 1e9, "pool3": 1e9, "pool4": 1e9}

    model_name = "pool4"
    textures_directory = "textures"
    all_textures = os.listdir(textures_directory)
    np.random.shuffle(all_textures)
    for texture in all_textures:
        print "Synthesizing texture", texture
        image_name = texture.split(".")[0]
        filename = textures_directory + "/" + texture
        img = np.load(filename)

        # Initialize texture synthesis
        text_synth = ts.TextureSynthesis(sess, my_model, img, layer_weights, model_name, image_name)

        # Do training
        text_synth.train()

        sys.stdout.flush()


if __name__ == "__main__":
    main()

