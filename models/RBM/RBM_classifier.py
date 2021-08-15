import numpy as np
from .RBM_base import RBM_Model
import tensorflow as tf
import torch
from tqdm import tqdm

class Pre_trainer:
	def __init__(self, n_visible, n_hidden, n_classes, lr=0.001, optim='adam', k=5, batch_size=32, epochs=100):
		self.n_hidden = n_hidden
		self.n_visible = n_visible
		self.lr = lr
		self.optim = optim
		self.k = k
		self.batch_size = batch_size
		self.epochs = epochs
		self.epsilon=1e-06
		self.n_classes = n_classes

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

	def get_torch_model(self):
		return torch.nn.Sequential(
			torch.nn.Linear(self.n_visible, self.n_hidden),
			torch.nn.Sigmoid(),
			torch.nn.Linear(self.n_hidden, self.n_classes),
			)

	def set_torch_model(self):
		model = self.get_torch_model()
		for param, wt in zip(model.parameters(), [self.W, self.hb.flatten()]):
			param.data = torch.from_numpy(wt)
		return model

	def fine_tune_torch(self, model, x, y):
		x = (x - np.min(x))/(np.max(x) - np.min(x) + self.epsilon)
		x = x.astype(np.float32)
		x = [x[index*self.batch_size:(index+1)*self.batch_size] for index in range(len(x)//self.batch_size)]
		y = [y[index*self.batch_size:(index+1)*self.batch_size] for index in range(len(y)//self.batch_size)]
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		for epoch in range(self.epochs):
			losses, accuracies = [], []
			bar = tqdm(enumerate(zip(x, y)), total=len(x))
			for batch_idx, (batch_x, target) in bar:
				batch_x = torch.from_numpy(batch_x)
				target = torch.from_numpy(target)
				optimizer.zero_grad()
				output = model(batch_x)
				loss = criterion(output, target)
				loss.backward()
				optimizer.step()
				preds = torch.argmax(output, 1)
				acc = torch.mean((preds==target).float())
				accuracies.append(acc.item())
				losses.append(loss.item())
				bar.set_description(str({'epoch': epoch, 'loss': round(sum(losses)/len(losses), 4), 'acc': round(sum(accuracies)/len(accuracies), 4)}))
				bar.refresh()
			bar.close()
		return model

	def get_tensorflow_model(self):
		# To Be Developed
		pass