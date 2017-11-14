import scipy.misc
import numpy as np

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def load_image(path):
    image = scipy.misc.imread(path)
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

if __name__ == "__main__":
    image = load_image("texture_images/pine_shoots.jpg")
    print image.shape

