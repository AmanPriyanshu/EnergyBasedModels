import numpy as np
from .DBN_base import DBN_Model
import tensorflow as tf
import torch
from tqdm import tqdm

class Pre_trainer:
	def __init__(self, n_visible, hidden_array, n_classes, lr=0.001, optim='adam', k=5, batch_size=32, epochs=100):
		self.hidden_array = hidden_array
		self.n_visible = n_visible
		self.lr = lr
		self.optim = optim
		self.k = k
		self.batch_size = batch_size
		self.epochs = epochs
		self.epsilon=1e-06
		self.n_classes = n_classes
		self.dbn = DBN_Model(n_visible, hidden_array, lr, optim, k, batch_size, epochs)

	def train_dbn(self, x):
		self.model_array = self.dbn.train(x)
		return self.model_array

	def get_torch_model(self):
		model_arr = []
		for idx, model in enumerate(self.model_array):
			model_arr.append(torch.nn.Linear(model['W'].shape[1], model['W'].shape[0]))
			model_arr.append(torch.nn.Sigmoid())
		model_arr.append(torch.nn.Linear(model['W'].shape[0], self.n_classes))
		return torch.nn.Sequential(*model_arr)
		
	def set_torch_model(self):
		params = []
		for model in self.model_array:
			params.append(model['W'])
			params.append(model['hb'].flatten())
		model = self.get_torch_model()
		for param, wt in zip(model.parameters(), params):
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