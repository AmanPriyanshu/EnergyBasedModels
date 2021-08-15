import numpy as np
from tqdm import tqdm, trange
from scipy.special import expit

class RBM_Model:
	def __init__(self, n_visible, n_hidden, lr=0.001, optim='adam', k=5, batch_size=32, epochs=10):
		self.n_hidden = n_hidden
		self.n_visible = n_visible
		self.lr = lr
		self.optim = optim
		self.k = k
		self.batch_size = batch_size
		self.epochs = epochs

		# Preparing for Adam

		self.beta_1=0.9
		self.beta_2=0.999
		self.epsilon=1e-07
		self.m = [0, 0, 0]
		self.v = [0, 0, 0]
		self.m_batches = {0:[], 1:[], 2:[]}
		self.v_batches = {0:[], 1:[], 2:[]}

		# Model Intialization

		std = 4 * np.sqrt(6. / (self.n_visible + self.n_hidden))
		self.W = np.random.normal(loc=0, scale=std, size=(self.n_hidden, self.n_visible)).astype(np.float32)
		self.vb = np.zeros(shape=(1, self.n_visible), dtype=np.float32)
		self.hb = np.zeros(shape=(1, self.n_hidden), dtype=np.float32)

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

	def adam(self, g, epoch, index):
		self.m[index] = self.beta_1 * self.m[index] + (1 - self.beta_1) * g
		self.v[index] = self.beta_2 * self.v[index] + (1 - self.beta_2) * np.power(g, 2)

		m_hat = self.m[index] / (1 - np.power(self.beta_1, epoch)) + (1 - self.beta_1) * g / (1 - np.power(self.beta_1, epoch))
		v_hat = self.v[index] / (1 - np.power(self.beta_2, epoch))
		return m_hat / (np.sqrt(v_hat) + self.epsilon)

	def update(self, v0, vk, ph0, phk, epoch):
		dW = (np.dot(v0.T, ph0).T - np.dot(vk.T, phk).T)
		dvb = np.expand_dims(np.sum((v0 - vk), 0), 0)
		dhb = np.expand_dims(np.sum((ph0 - phk), 0), 0)

		if self.optim == 'adam':
			dW = self.adam(dW, epoch, 0)
			dvb = self.adam(dvb, epoch, 1)
			dhb = self.adam(dhb, epoch, 2)

		self.W -= self.lr * dW
		self.vb -= self.lr * dvb
		self.hb -= self.lr * dhb

	def single_epoch(self, x, epoch, disable=False):
		train_loss = 0
		bar = tqdm(enumerate(x), total=len(x), disable=disable)
		for batch_idx, batch_x in bar:
			v0 = batch_x
			vk = batch_x
			ph0, _ = self.sample_h(v0)
			for k in range(self.k):
				_, hk = self.sample_h(vk)
				_, vk = self.sample_v(hk)
			phk, _ = self.sample_h(vk)
			self.update(v0, vk, ph0, phk, epoch+1)
			train_loss += np.mean(np.abs(v0-vk))
			bar.set_description(str({'epoch': epoch+1, 'loss': round(train_loss/(batch_idx+1), 4)}))
			bar.refresh()
		bar.close()
		return train_loss/(batch_idx+1)

	def train(self, x, disable_internal=False, disable_external=True, return_loss=False, part_of_dbn=False):
		x = x.astype(np.float32)
		if not part_of_dbn:
			x = (x - np.min(x))/(np.max(x) - np.min(x) + self.epsilon)
		x = [x[index*self.batch_size:(index+1)*self.batch_size] for index in range(len(x)//self.batch_size)]
		bar = trange(self.epochs, disable=disable_external)
		for epoch in bar:
			mae = self.single_epoch(x, epoch, disable=disable_internal)
			bar.set_description(str({'epoch': epoch+1, 'loss': round(mae, 4)}))
			bar.refresh()
		bar.close()
		model = {'W': self.W, 'hb': self.hb, 'vb': self.vb}
		if return_loss:
			return model, mae
		else:
			return model