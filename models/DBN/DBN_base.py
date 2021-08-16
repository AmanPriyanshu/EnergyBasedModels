from ..RBM.RBM_base import RBM_Model
import numpy as np
from scipy.special import expit

class DBN_Model:
	def __init__(self, n_visible, hidden_array, lr=0.001, optim='adam', k=5, batch_size=32, epochs=10):
		self.hidden_array = [n_visible]+hidden_array
		self.n_visible = n_visible
		self.lr = lr
		self.optim = optim
		self.k = k
		self.batch_size = batch_size
		self.epochs = epochs
		self.model_array = []
		self.epsilon = 1e-6

	def sample_h(self, x, model):
		z = np.dot(x, model['W'].T) + model['hb']
		p_h_given_v = expit(-z)
		random_sample = np.random.uniform(low=0.0, high=1.0, size=p_h_given_v.shape)
		sampled_h = (random_sample<p_h_given_v).astype(np.float32)
		return p_h_given_v, sampled_h

	def get_hidden_samples(self, x, index):
		if index==0:
			return x
		hidden_features = x
		for model in self.model_array:
			for idx, batch_x in enumerate([hidden_features[index*self.batch_size:(index+1)*self.batch_size] for index in range(len(hidden_features)//self.batch_size)]):
				features, _ = self.sample_h(batch_x, model)
				if idx==0:
					hidden_features = features
				else:
					hidden_features = np.concatenate((hidden_features, features), 0)
		return hidden_features

	def train(self, x):
		x = (x - np.min(x))/(np.max(x) - np.min(x) + self.epsilon)
		for index in range(len(self.hidden_array)-1):
			print("Layer:", index+1, "n_visible:", self.hidden_array[index], "n_hidden:", self.hidden_array[index+1])
			rbm = RBM_Model(self.hidden_array[index], self.hidden_array[index+1], lr=self.lr, optim=self.optim, k=self.k, batch_size=self.batch_size, epochs=self.epochs)
			model, loss = rbm.train(self.get_hidden_samples(x, index), disable_internal=True, disable_external=False, return_loss=True, part_of_dbn=True)
			self.model_array.append(model)
		return self.model_array