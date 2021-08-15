import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from examples.data import get_data
from models.RBM.RBM_classifier import Pre_trainer as Pre_trainer_classifier
from models.RBM.RBM_synthetic_generator import Pre_trainer as Pre_trainer_synthetic_generator
from matplotlib import pyplot as plt
import numpy as np

def RBM_synthetic_generation_example(labels, images):
	images_reshaped = images.reshape((len(images), -1))
	rbm = Pre_trainer_synthetic_generator(n_visible=784, n_hidden=10, epochs=15, optim='adam')
	rbm.train_rbm(images_reshaped)
	synthetic_images, hidden_features = rbm.get_synthetic_data(images_reshaped)
	synthetic_images = np.reshape(synthetic_images, (-1, 28, 28))
	hidden_features = np.reshape(hidden_features, (-1, 5, 2))
	plt.cla()
	fig = plt.figure(figsize = (15,60))
	for digit in range(10):
		index = np.argwhere(labels==digit).flatten()[0]
		plt.subplot(10, 3, digit*3+1)
		plt.axis('off')
		plt.imshow(images[index], cmap='gray')
		plt.subplot(10, 3, digit*3+2)
		plt.axis('off')
		plt.imshow(hidden_features[index], cmap='gray')
		plt.subplot(10, 3, digit*3+3)
		plt.axis('off')
		plt.imshow(synthetic_images[index], cmap='gray')
	fig.tight_layout()
	plt.savefig('RBM_synthetic_generations.png')

def RBM_classification_example(labels, images):
	images_reshaped = images.reshape((len(images), -1))
	rbm = Pre_trainer_classifier(n_visible=784, n_hidden=400, n_classes=10, epochs=15, optim='adam')
	rbm.train_rbm(images_reshaped)
	print("------Without Pre Training------")
	model = rbm.get_torch_model()
	rbm.fine_tune_torch(model, images_reshaped, labels)
	print("-------With Pre Training--------")
	model = rbm.set_torch_model()
	rbm.fine_tune_torch(model, images_reshaped, labels)
	
if __name__ == '__main__':
	data = get_data("./data/train.csv")
	#RBM_synthetic_generation_example(data['labels'], data['images'])
	RBM_classification_example(data['labels'], data['images'])