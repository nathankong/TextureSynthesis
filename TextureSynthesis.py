import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from ImageUtils import *
from model import *

SAVE_STEP = 750
ITERATIONS = 2251

class TextureSynthesis:
    def __init__(self, sess, model, actual_image, layer_constraints, model_name, image_name):
        # 'layer_constraints' is dictionary with key = VGG layer and value = weight (w_l)
        # 'sess' is tensorflow session
        self.model_name = model_name # Of the form: conv#
        self.image_name = image_name # Of the form: imageName

        self.sess = sess
        self.sess.run(tf.initialize_all_variables())

        self.model = model # Model instance
        assert self.model.model_initialized(), "Model not created yet."
        self.model_layers = self.model.get_model()

        # Layer weights for the loss function
        self.layer_weights = layer_constraints

        self.actual_image = actual_image # 256x256x3

        self.init_image = self._gen_noise_image()
        self.constraints = self._get_constraints() # {layer_name: activations}

    def get_texture_loss_2(self, desired, actual):
        layer_activations_shape = desired.shape
        num_filters = layer_activations_shape[3] # N
        num_spatial_locations = layer_activations_shape[1] * layer_activations_shape[2] # M

        desired_gram_matrix = self._compute_gram_matrix_np(desired, num_filters, num_spatial_locations)
        layer_gram_matrix = self._compute_gram_matrix(actual, num_filters, num_spatial_locations)

        total_loss = (1.0 / (4 * (num_filters**2) * (num_spatial_locations**2))) \
                          * tf.reduce_sum(tf.pow(desired_gram_matrix - layer_gram_matrix, 2))
        return total_loss

    def get_texture_loss(self):
        total_loss = 0.0
        for layer in self.layer_weights.keys():
            layer_activations = self.model_layers[layer]
            layer_activations_shape = layer_activations.get_shape().as_list()
            assert len(layer_activations_shape) == 4 # (1, H, W, outputs)
            assert layer_activations_shape[0] == 1, "Only supports 1 image at a time."
            num_filters = layer_activations_shape[3] # N
            num_spatial_locations = layer_activations_shape[1] * layer_activations_shape[2] # M
            layer_gram_matrix = self._compute_gram_matrix(layer_activations, num_filters, num_spatial_locations)
            desired_gram_matrix = self.constraints[layer]

            total_loss += self.layer_weights[layer] * (1.0 / (4 * (num_filters**2) * (num_spatial_locations**2))) \
                          * tf.reduce_sum(tf.pow(desired_gram_matrix - layer_gram_matrix, 2))
        return total_loss

    def _get_constraints(self):
        self.sess.run(tf.initialize_all_variables())
        constraints = dict()
        for layer in self.layer_weights:
            self.sess.run(self.model_layers['input'].assign(self.actual_image))
            layer_activations = self.sess.run(self.model_layers[layer])
            num_filters = layer_activations.shape[3] # N
            num_spatial_locations = layer_activations.shape[1] * layer_activations.shape[2] # M
            constraints[layer] = self._compute_gram_matrix_np(layer_activations, num_filters, num_spatial_locations)

        return constraints

    def _compute_gram_matrix_np(self, F, N, M):
        F = F.reshape(M, N)
        return np.dot(F.T, F)

    def _compute_gram_matrix(self, F, N, M):
        # F: (1, height, width, num_filters), layer activations
        # N: num_filters
        # M: number of spatial locations in filter (filter size ** 2)
        F = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(F), F)

    def _gen_noise_image(self):
        input_size = self.model_layers["input"].get_shape().as_list()
        return np.random.randn(input_size[0], input_size[1], input_size[2], input_size[3])

    def train(self):
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.model_layers["input"].assign(self.actual_image))
        #E = [self.layer_weights[layer] * self.get_texture_loss_2(self.sess.run(self.model_layers[layer]), self.model_layers[layer]) for layer in self.layer_weights]

        #content_loss = sum(E)
        content_loss = self.get_texture_loss()
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(content_loss)

        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.model_layers["input"].assign(self.init_image))
        for i in range(ITERATIONS):
            self.sess.run(train_step)
            if i % 1 == 0:
                print "Iteration: " + str(i) + "; Loss: " + str(self.sess.run(content_loss))
            if i % SAVE_STEP == 0:
                print "Saving image..."
                curr_img = self.sess.run(self.model_layers["input"])
                filename = "output/%s_%s_step_%d" % (self.model_name, self.image_name, i)
                save_image(filename, curr_img)
            sys.stdout.flush()

    def train2(self, train_step, loss_func):
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.model_layers["input"].assign(self.init_image))
        for i in range(ITERATIONS):
            self.sess.run(train_step)
            if i % 1 == 0:
                print "Iteration: " + str(i) + "; Loss: " + str(sess.run(loss_func))

#===============================
def gram_matrix(x, area, depth):
    x1 = tf.reshape(x,(area,depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g

def gram_matrix_val(x, area, depth):
    x1 = x.reshape(area,depth)
    g = np.dot(x1.T, x1)
    return g

def build_style_loss(a, x):
    M = a.shape[1]*a.shape[2]
    N = a.shape[3]
    A = gram_matrix_val(a, M, N )
    G = gram_matrix(x, M, N )
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
    return loss
#===============================



if __name__ == "__main__":
    vgg_weights = VGGWeights('vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()

    img = np.load("texture_images/pine_shoots_norm.npy")

    sess = tf.Session()
    layer_weights = {"conv1_1": 1e9, "conv2_1": 1e9, "conv3_1": 1e9}
    model_name = "conv3"
    image_name = "pine_shoots"
    text_synth = TextureSynthesis(sess, my_model, img, layer_weights, model_name, image_name)

    print "Success in initializing."
    print "Training..."

#    #STYLE_LAYERS=[('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]
#    STYLE_LAYERS=[('pool1',1.),('pool2',1.),('pool3',1.),('pool4',1.)]
#    sess.run([text_synth.model_layers['input'].assign(img)])
#    cost_style = sum(map(lambda l: l[1]*build_style_loss(sess.run(text_synth.model_layers[l[0]]), text_synth.model_layers[l[0]])
#                    , STYLE_LAYERS))
#
#    cost_total = cost_style
#    optimizer = tf.train.AdamOptimizer(2.0)
#
#    train_step = optimizer.minimize(cost_total)
#
#    text_synth.train2(train_step, cost_total)

    text_synth.train()

