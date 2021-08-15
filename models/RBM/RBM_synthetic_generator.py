import numpy as np
from .RBM_base import RBM_Model
from scipy.special import expit
from tqdm import tqdm

class Pre_trainer:
	def __init__(self, n_visible, n_hidden, lr=0.001, optim='adam', k=5, batch_size=32, epochs=100):
		self.n_hidden = n_hidden
		self.n_visible = n_visible
		self.lr = lr
		self.optim = optim
		self.k = k
		self.batch_size = batch_size
		self.epochs = epochs
		self.epsilon=1e-06

		self.rbm = RBM_Model(n_visible, n_hidden, lr, optim, k, batch_size, epochs)
		self.model, self.W, self.vb, self.hb = None, None, None, None

	def train_rbm(self, x):
		self.model = self.rbm.train(x)
		self.set_model(self.model)
		return self.model

	def set_model(self, model):
		self.W = model['W']
		self.vb = model['vb']
		self.hb = model['hb']

	def sample_h(self, x):
		z = np.dot(x, self.W.T) + self.hb
		p_h_given_v = expit(-z)
		random_sample = np.random.uniform(low=0.0, high=1.0, size=p_h_given_v.shape)
		sampled_h = (random_sample<p_h_given_v).astype(np.float32)
		return p_h_given_v, sampled_h

	def sample_v(self, y):
		z = np.dot(y, self.W) + self.vb
		p_v_given_h = expit(-z)
		random_sample = np.random.uniform(low=0.0, high=1.0, size=p_v_given_h.shape)
		sampled_v = (random_sample<p_v_given_h).astype(np.float32)
		return p_v_given_h, sampled_v

	def get_synthetic_data(self, x):
		x = x.astype(np.float32)
		x = (x - np.min(x))/(np.max(x) - np.min(x) + self.epsilon)
		x = [x[index*self.batch_size:(index+1)*self.batch_size] for index in range(len(x)//self.batch_size)]

		for batch_idx, batch_x in tqdm(enumerate(x), desc="Generating", total=len(x)):
			v0 = batch_x
			vk = batch_x
			ph0, h0 = self.sample_h(v0)
			v_array, h_array = [], []
			for k in range(self.k):
				_, hk = self.sample_h(vk)
				_, vk = self.sample_v(hk)
				v_array.append(vk)
				h_array.append(hk)
			phk, _ = self.sample_h(vk)
			synthetic_hidden_features_batch = np.mean(np.stack(h_array), 0)
			synthetic_visible_features_batch = np.mean(np.stack(v_array), 0)
			if batch_idx==0:
				synthetic_visible_features = synthetic_visible_features_batch
				synthetic_hidden_features = synthetic_hidden_features_batch
			else:
				synthetic_hidden_features = np.concatenate((synthetic_hidden_features, synthetic_hidden_features_batch), 0)
				synthetic_visible_features = np.concatenate((synthetic_visible_features, synthetic_visible_features_batch), 0)
		return synthetic_visible_features, synthetic_hidden_features