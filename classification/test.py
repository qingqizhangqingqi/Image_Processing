import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from se_resnext import seresnext50_32x4d
#from resnest import resnest50_fast_1s4x24d, resnest50_fast_4s2x40d
from torch.utils.data import Dataset, DataLoader
import argparse
from dataset import TestDataset
from se_resnext import seresnext50_32x4d
import csv

def parse_args():
	parser = argparse.ArgumentParser(description='inference')
	parser.add_argument('--gpu', type=str, default='0', help='gpu id')
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--model', type=str, default='models/epoch_20.pth')
	args = parser.parse_args()
	return args

def to_tensor(buff):
	return buff.permute(0, 3, 1, 2)

def get_categories(file_path):
	category_map = {}
	with open(file_path) as fp:
		reader = csv.reader(fp)
		for index, row in enumerate(reader):
			if index == 0:
				continue

			category_map[row[1]] = row[0]
	return category_map			

@torch.no_grad()
def test(model_path, batch_size):
	current_epoch = 0
	
	width = height = 320
	data_dir = 'data/data/'
	dataset = TestDataset(data_dir, 'data/annotation.csv', width, height)
	image_count = len(dataset)
	category_map = get_categories('data/species.csv')

	model = seresnext50_32x4d(False, num_classes=len(category_map)).cuda()
	#model = resnest50_fast_1s4x24d(num_classes=len(category_map)).cuda()
	#model = resnest50_fast_4s2x40d(num_classes=len(category_map)).cuda()
	model_dict = torch.load(model_path).module.state_dict()
	model.load_state_dict(model_dict)
	model.eval()

	correct = 0
	dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	for i, data in enumerate(dataloader):
		im = to_tensor(data[0]).cuda()
		category_id = data[1].cuda()
		cate_fc = model(im)
		probs = torch.nn.Softmax(dim=1)(cate_fc)
		cls_id = torch.max(probs, 1)[1]
		correct += (cls_id == category_id).sum().item()
	print("correct = {}, total = {}, {}".format(correct, image_count, correct / image_count))

def main():
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	gpus = len(args.gpu.split(','))
	batch_size = args.batch_size
	batch_size = batch_size * gpus
	test(args.model, batch_size)

if __name__ == "__main__":
	main()
