import pickle
import numpy as np

values = pickle.load(open('vgg19_normalized.pkl'))['param values']

for i in range(len(values)):
	print values[i].shape
