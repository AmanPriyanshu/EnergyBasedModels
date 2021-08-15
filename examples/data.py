import pandas as pd
import numpy as np

def get_data(path, seed=0):
	df = pd.read_csv(path)
	df = df.values
	np.random.seed(seed)
	np.random.shuffle(df)
	labels = df.T[0]
	images = df.T[1:].T
	images = np.reshape(images, (-1, 28, 28))
	return {'labels': labels, 'images': images}