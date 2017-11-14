import pickle

class VGGWeights:
    """ Defines the class that contains the function to obtain weights
        from a file. Weight shape: [out_channels, in_channels, H, W]
    """

    def __init__(self, weights_file):
        self.weights_file = weights_file


    def get_normalized_vgg_weights(self):
        values = pickle.load(open(self.weights_file))['param values']
        return values

