import numpy as np
from .RBM_base import RBM_Model

class Pre_trainer:
	def __init__(self, n_visible, n_hidden, lr=0.001, optim='adam', k=5, batch_size=32, epochs=100):
		self.n_hidden = n_hidden
		self.n_visible = n_visible
		self.lr = lr
		self.optim = optim
		self.k = k
		self.batch_size = batch_size
		self.epochs = epochs

		self.rbm = RBM_Model(n_visible, n_hidden, lr, optim, k, batch_size, epochs)
		self.model = None

	def train_rbm(self, x):
		self.model = self.rbm.train(x)
		return self.model