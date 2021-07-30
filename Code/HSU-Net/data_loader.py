import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import MNISTreader
import Dataset_reader

class ImageFolder(data.Dataset):
	def __init__(self,mode='train'):
		self.image1,self.image2,self.labels = Dataset_reader.get_dataset(mode)
		print(f"len:{len(self.image2)}")
		print(torch.tensor(self.image2).shape)

	def __getitem__(self,index):

		return torch.from_numpy(np.array(self.image1[index])).float(), \
			   torch.from_numpy(np.array(self.image2[index])).float(),\
			   torch.from_numpy(np.array(self.labels[index])).long()
		#return torch.from_numpy(np.array([self.image[index]])).float(),torch.from_numpy(np.array(self.labels[index])).long()
	def __len__(self):
		return len(self.labels)

def get_loader(batch_size, num_workers=2, mode='train'):
	dataset = ImageFolder(mode=mode)

	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers,
								  drop_last=False)
	return data_loader
