import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net,HSU_Net
import csv
import cfg
from sklearn import metrics
import random

np.random.seed(1234)
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarlk = False

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
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=3,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=3,output_ch=self.output_ch)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=self.img_ch,output_ch=self.output_ch,t=self.t)
			# device_ids = [i for i in range(int(torch.cuda.device_count()))]
			# print(torch.cuda.device_count())
			self.unet = torch.nn.DataParallel(self.unet.cuda())
		elif self.model_type == 'HSU_Net':
			self.unet = HSU_Net(img1_ch=self.img1_ch,img2_ch=self.img2_ch, output_ch=self.output_ch, t=self.t)
			if torch.cuda.is_available():
				print("using cuda!")
				# self.unet = self.unet.cuda('cuda:1') #ProtoNet
				# print(next(self.unet.parameters()).device)
				if torch.cuda.device_count() >= 2:
					print("using devices num {}".format(torch.cuda.device_count()))
					self.unet=torch.nn.DataParallel(self.unet).cuda()
			# print(self.unet.device)

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr,
									(self.beta1, self.beta2))
		# self.unet = self.unet.to(self.device)
		print(next(self.unet.parameters()).device)
		# self.print_network(self.unet, self.model_type)

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
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			best_epoch = 0

			for epoch in range(self.num_epochs):
				self.unet = self.unet.float()
				self.unet.train(True)
				Loss = []
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				PR_all = np.array([])
				lable_all = np.array([])

				for i, data in enumerate(self.train_loader):
					# GT : Ground Truth
					images1 = data[0]
					images2 = data[1]
					GT = data[2]

					# images1 = images1.to(self.device)
					# images2 = images2.to(self.device)
					# GT = GT.to(self.device)
					if torch.cuda.is_available():
						images1 = images1.cuda()
						images2 = images2.cuda()
						GT = GT.cuda()

					# NR : Net Result
					NR = self.unet(images1,images2)

					loss = self.criterion(NR,GT).sum()

					if abs(loss.item()-1.609437942504) < 1e-6:
						loss +=1

					print(i,"loss:",loss.item())
					Loss.append(loss.item())

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					acc += get_accuracy(NR,GT)

					PR_all = np.append(PR_all, NR.cpu().argmax(dim=1))
					lable_all = np.append(lable_all, GT.cpu())
					# print(PR_all)
					# print(lable_all)

					#SE += get_sensitivity(SR,GT)
					#SP += get_specificity(SR,GT)
					#PC += get_precision(SR,GT)
					#F1 += get_F1(SR,GT)
					#JS += get_JS(SR,GT)
					#DC += get_DC(SR,GT)
					length += len(GT)
				
				torch.cuda.empty_cache()

				PR_all = PR_all.astype(np.int16)
				lable_all = lable_all.astype(np.int16)

				acc = acc/length
				oa = metrics.accuracy_score(lable_all, PR_all)
				recall = metrics.recall_score(lable_all, PR_all, average="macro")
				precision = metrics.precision_score(lable_all, PR_all, average="macro")
				F1 = metrics.f1_score(lable_all, PR_all, average="macro")
				print("TRAIN acc:",acc)
				'''
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				'''
				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, OA: %.4f, recall: %.4f, precision: %.4f, F1: %.4f' % (
					  epoch+1, self.num_epochs, np.mean(Loss),acc,oa,recall,precision,F1))

				# Decay learning rate
				'''if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))'''
				
				
				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0

				PR_all = np.array([])
				lable_all = np.array([])

				print("valid_loader:",len(self.valid_loader))
				
				with torch.no_grad():
					for i, (images1,images2, GT) in enumerate(self.valid_loader):
						# images1 = images1.to(self.device)
						# images2 = images2.to(self.device)
						# GT = GT.to(self.device)

						if torch.cuda.is_available():
							images1 = images1.cuda()
							images2 = images2.cuda()
							GT = GT.cuda()

						#SR = F.sigmoid(self.unet(images))
						SR = self.unet(images1,images2)
						acc += get_accuracy(SR,GT)

						PR_all = np.append(PR_all, SR.cpu().argmax(dim=1))
						lable_all = np.append(lable_all, GT.cpu())

						#SE += get_sensitivity(SR,GT)
						#SP += get_specificity(SR,GT)
						#PC += get_precision(SR,GT)
						#F1 += get_F1(SR,GT)
						#JS += get_JS(SR,GT)
						#DC += get_DC(SR,GT)
						length += len(GT)
					
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				unet_score = acc

				PR_all = PR_all.astype(np.int16)
				lable_all = lable_all.astype(np.int16)

				oa = metrics.accuracy_score(lable_all, PR_all)
				recall = metrics.recall_score(lable_all, PR_all, average="macro")
				precision = metrics.precision_score(lable_all, PR_all, average="macro")
				F1 = metrics.f1_score(lable_all, PR_all, average="macro")

				print('[Validation] Acc: %.4f, OA: %.4f, recall: %.4f, precision: %.4f, F1: %.4f'%(acc,oa,recall,precision,F1))
				print('Best %s model score : %.4f'%(best_epoch, best_unet_score))

				# Save Best U-Net model
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					self.best_epoch = epoch
					best_epoch = epoch
					print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
					# best_unet = self.unet.state_dict()
					# torch.save(best_unet,unet_path)

				torch.cuda.empty_cache()

			#===================================== Test ====================================#

			# self.build_model()
			# torch.cuda.empty_cache()
			# self.unet.load_state_dict(torch.load(unet_path))

			print('test datasets!')

			self.unet.train(False)

			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			PR_all = np.array([])
			lable_all = np.array([])

			with torch.no_grad():
				for i, (images1,images2, GT) in enumerate(self.valid_loader):

					# images1 = images1.to(self.device)
					# images2 = images2.to(self.device)
					# GT = GT.to(self.device)

					if torch.cuda.is_available():
						images1 = images1.to(self.device)
						images2 = images2.to(self.device)
						GT = GT.to(self.device)

					NR = self.unet(images1,images2)
					acc += get_accuracy(NR,GT)

					PR_all = np.append(PR_all, NR.cpu().argmax(dim=1))
					lable_all = np.append(lable_all, GT.cpu())
					#SE += get_sensitivity(SR,GT)
					#SP += get_specificity(SR,GT)
					#PC += get_precision(SR,GT)
					#F1 += get_F1(SR,GT)
					#JS += get_JS(SR,GT)
					#DC += get_DC(SR,GT)

					length += len(GT)

				acc = acc/length
				#SE = SE/length
				#SP = SP/length
				#PC = PC/length
				#F1 = F1/length
				#JS = JS/length
				#DC = DC/length
				#unet_score = JS + DC

				PR_all = PR_all.astype(np.int16)
				lable_all = lable_all.astype(np.int16)

				oa = metrics.accuracy_score(lable_all, PR_all)
				recall = metrics.recall_score(lable_all, PR_all, average="macro")
				precision = metrics.precision_score(lable_all, PR_all, average="macro")
				F1 = metrics.f1_score(lable_all, PR_all, average="macro")

				print('[Validation] Acc: %.4f, OA: %.4f, recall: %.4f, precision: %.4f, F1: %.4f'%(acc,oa,recall,precision,F1))

				# f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
				# wr = csv.writer(f)
				# wr.writerow([self.model_type,acc.item(),oa,recall,precision,F1,self.batch_size,self.lr,self.best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
				# f.close()