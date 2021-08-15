import numpy as np
from .DBN_base import DBN_Model
from scipy.special import expit
from tqdm import tqdm

class Pre_trainer:
	def __init__(self, n_visible, hidden_array, lr=0.001, optim='adam', k=5, batch_size=32, epochs=10):
		self.hidden_array = hidden_array
		self.n_visible = n_visible
		self.lr = lr
		self.optim = optim
		self.k = k
		self.batch_size = batch_size
		self.epochs = epochs
		self.epsilon=1e-06
		self.dbn = DBN_Model(n_visible, hidden_array, lr, optim, k, batch_size, epochs)

	def train_dbn(self, x):
		self.model_array = self.dbn.train(x)
		return self.model_array

	def sample_h(self, x, model):
		z = np.dot(x, model['W'].T) + model['hb']
		p_h_given_v = expit(-z)
		random_sample = np.random.uniform(low=0.0, high=1.0, size=p_h_given_v.shape)
		sampled_h = (random_sample<p_h_given_v).astype(np.float32)
		return p_h_given_v, sampled_h

	def sample_v(self, y, model):
		z = np.dot(y, model['W']) + model['vb']
		p_v_given_h = expit(-z)
		random_sample = np.random.uniform(low=0.0, high=1.0, size=p_v_given_h.shape)
		sampled_v = (random_sample<p_v_given_h).astype(np.float32)
		return p_v_given_h, sampled_v

	def get_hidden(self, x):
		hidden_features = x
		for model in self.model_array:
			for idx, batch_x in enumerate([hidden_features[index*self.batch_size:(index+1)*self.batch_size] for index in range(len(hidden_features)//self.batch_size)]):
				features = []
				for _ in range(self.k):
					_, v = self.sample_h(batch_x, model)
					features.append(v)
				features = np.mean(np.stack(features), 0)
				if idx==0:
					hidden_features = features
				else:
					hidden_features = np.concatenate((hidden_features, features), 0)
		return hidden_features

	def get_visible(self, y):
		hidden_features = y
		for model in self.model_array[::-1]:
			for idx, batch_x in enumerate([hidden_features[index*self.batch_size:(index+1)*self.batch_size] for index in range(len(hidden_features)//self.batch_size)]):
				features = []
				for _ in range(self.k):
					_, v = self.sample_v(batch_x, model)
					features.append(v)
				features = np.mean(np.stack(features), 0)
				if idx==0:
					hidden_features = features
				else:
					hidden_features = np.concatenate((hidden_features, features), 0)
		return hidden_features

	def get_synthetic_data(self, x):
		x = x.astype(np.float32)
		x = (x - np.min(x))/(np.max(x) - np.min(x) + self.epsilon)
		x = [x[index*self.batch_size:(index+1)*self.batch_size] for index in range(len(x)//self.batch_size)]

		for batch_idx, batch_x in tqdm(enumerate(x), desc="Generating", total=len(x)):
			synthetic_hidden_features_batch = self.get_hidden(batch_x)
			synthetic_visible_features_batch = self.get_visible(synthetic_hidden_features_batch)
			if batch_idx==0:
				synthetic_visible_features = synthetic_visible_features_batch
				synthetic_hidden_features = synthetic_hidden_features_batch
			else:
				synthetic_hidden_features = np.concatenate((synthetic_hidden_features, synthetic_hidden_features_batch), 0)
				synthetic_visible_features = np.concatenate((synthetic_visible_features, synthetic_visible_features_batch), 0)
		return synthetic_visible_features, synthetic_hidden_features