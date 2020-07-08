import os
import cv2
import time
import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import numpy as np
from se_resnext import seresnext50_32x4d
from resnest import resnest50_fast_1s4x24d, resnest50_fast_4s2x40d
from torch.utils.data import Dataset, DataLoader
import argparse
from dataset import ImageDataset
from mixup import mixup_data, mixup_criterion
import csv
import random
from torch.autograd import Variable

def parse_args():
	parser = argparse.ArgumentParser(description='inference')
	parser.add_argument('--gpu', type=str, default='0', help='gpu id')
	parser.add_argument('--batch_size', type=int, default=11)
	parser.add_argument('--resume_from', type=str, default=None)
	args = parser.parse_args()
	return args

def to_tensor(buff):
	return buff.permute(0, 3, 1, 2)

def get_epoch(model_path):
	epoch = model_path.split('.')[0].split('_')[1]
	return int(epoch)

def get_categories(file_path):
	category_map = {}
	with open(file_path) as fp:
		reader = csv.reader(fp)
		for index, row in enumerate(reader):
			if index == 0:
				continue

			category_map[row[1]] = row[0]
	return category_map			

def train(args, batch_size):
	current_epoch = 0
	
	width = height = 320
	data_dir = 'data/data/'
	dataset = ImageDataset(data_dir, 'data/training.csv', width, height)
	category_map = get_categories('data/species.csv')

	file_count = len(dataset) // batch_size
	model = seresnext50_32x4d(True, num_classes=len(category_map), drop_rate=0)
	#model = resnest50_fast_1s4x24d('pretrained/resnest50_fast_1s4x24d-d4a4f76f.pth', num_classes=len(category_map))
	#model = resnest50_fast_4s2x40d('pretrained/resnest50_fast_4s2x40d-41d14ed0.pth', num_classes=len(category_map))
	model = torch.nn.DataParallel(model).cuda()
	if args.resume_from != None:
		current_epoch = get_epoch(args.resume_from)
		model_dict = torch.load(args.resume_from).module.state_dict()
		model.module.load_state_dict(model_dict)
		print("resume from ", args.resume_from)

	lr = 0.005
	use_lr = lr
	optimizer = SGD(model.parameters(), lr=use_lr, momentum=0.9, weight_decay=0.0001)

	print(current_epoch)	
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.5, last_epoch=-1)
	#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00005)
	#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)

	epochs = 15
	loss_fn = nn.CrossEntropyLoss()
	show_loss_loop = 10
	alpha = 0.1
	for epoch in range(current_epoch, epochs):
		show_cate_loss = 0
		dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

		start = time.time()
		for i, data in enumerate(dataloader):
			optimizer.zero_grad()
			if random.random() < 0.5:
				inputs = to_tensor(data[0]).cuda()
				targets = data[1].cuda()
				inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, True)
				inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
				outputs = model(inputs)
				cate_loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)
			else:
				im = to_tensor(data[0]).cuda()
				category_id = data[1].cuda()
				cate_fc = model(im)
				cate_loss = loss_fn(cate_fc, category_id)

			cate_loss.backward()
			optimizer.step()

			show_cate_loss += cate_loss.item()
			if (i+1) % show_loss_loop == 0:
				end = time.time()
				use_time = (end - start) / show_loss_loop
				start = end
				need_time = ((file_count * (epochs - epoch) - i) * use_time) / 60 / 60

				show_cate_loss /= show_loss_loop
				print("epoch: {}/{} iter:{}/{} lr:{:.5f}, cate_loss:{:.5f}, use_time:{:.2f}/iter, need_time:{:.2f} h".\
					format(epoch+1, epochs, (i+1), file_count, lr_scheduler.get_lr()[0], show_cate_loss, use_time, need_time))
				show_cate_loss = 0

		lr_scheduler.step()
		torch.save(model, 'models/epoch_{}.pth'.format(epoch+1))

def main():
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	gpus = len(args.gpu.split(','))
	batch_size = args.batch_size
	batch_size = batch_size * gpus
	train(args, batch_size)

if __name__ == "__main__":
	main()
