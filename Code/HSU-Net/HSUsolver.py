import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
from evaluation import *
from network import HSU_Net
import csv
import cfg

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img1_ch = config.img1_ch
		self.img2_ch = config.img2_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.CrossEntropyLoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
        self.unet = HSU_Net(img1_ch=self.img1_ch,img2_ch=self.img2_ch, output_ch=self.output_ch, t=self.t)
        self.unet = torch.nn.DataParallel(self.unet).cuda()
		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr,
									(self.beta1, self.beta2))
		self.unet = self.unet.to(self.device)
       
	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			#-param_group['lr'] = lr
			param_group['lr'] = self.lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()
		#self.optimizer.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img

	def train(self):
		"""Train encoder, generator and discriminator."""
		#====================================== Training ===========================================#
		#===========================================================================================#
		unet_path = os.path.join(cfg.model_save_path,f"HSUnet_{self.batch_size}_{self.num_epochs}_{self.lr}_{self.num_epochs_decay}.pkl")
		# U-Net Train
		if os.path.isfile(unet_path) and False:
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))