import argparse
import os
import cv2
import torch
from mmdet.apis import inference_detector, init_detector, show_result
import json
import numpy as np
from mmdet.ops.nms.nms_wrapper import nms
from torch.utils.data import Dataset, DataLoader
import time

class MyDataset(Dataset):
	def __init__(self, data_dir):
		data_list = os.listdir(data_dir)
		self.data_list = []
		for data in data_list:
			if not os.path.isdir(os.path.join(data_dir, data)):
				continue

			self.data_list.append(data)
		
		self.data_dir = data_dir

	def __getitem__(self, index):
		image_dir = self.data_list[index]
		image_path = os.path.join(self.data_dir, image_dir, image_dir + '.jpg')
		style = image_dir.split('_')[0]
		template_path = os.path.join(self.data_dir, image_dir, 'template_{}.jpg'.format(style))
		im = cv2.imread(image_path)
		template = cv2.imread(template_path)
		return image_dir, im, template


	def __len__(self):
		return len(self.data_list)


def parse_args():
	parser = argparse.ArgumentParser(description='inference')
	parser.add_argument('--config', type=str, help='test config file path')
	parser.add_argument('--model', type=str, help='model file path')
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--data', type=str, help='data dir')
	args = parser.parse_args()
	return args

def get_cls_type(index):
	index_map = {'0':1, '1':2, '2':3, '3':4, '4':5, '5':6, '6':7, '7':8, '8':9, '9':10, '10':11, '11':12, '12':13, '13':14, '14':15}

	index = str(index)
	return index_map[index]

def main():
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	model = init_detector(args.config, args.model, device=torch.device('cuda', args.gpu))

	#root_dir = "/tcdata/guangdong1_round2_testB_20191024"
	root_dir = args.data
	results = []

	dataset = MyDataset(root_dir)
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
	count = len(dataloader)
	start = time.time()
	for image_index, data in enumerate(dataloader):
		image_dir, img, template_img = data[0][0], data[1][0], data[2][0]
		img, template_img = img.numpy(), template_img.numpy()
		height, width, _ = img.shape
		sub_height, sub_width = int(height/2), int(width/2)
		dets = []
		for row in range(2):
			for col in range(2):
				height_start = row * sub_height
				height_end = (row+1) * sub_height
				width_start = col * sub_width
				width_end = (col+1) * sub_width
				sub_im = img[height_start:height_end, width_start:width_end, :]
				sub_template = template_img[height_start:height_end, width_start:width_end, :]
				sub_dets = inference_detector(model, sub_im, sub_template, 0.065)
				for index, sub_det in enumerate(sub_dets):
					for sub_item in sub_det:
						sub_item[0] += width_start
						sub_item[1] += height_start
						sub_item[2] += width_start
						sub_item[3] += height_start
						sub_item[4] /= 8

				if len(dets) == 0:
					for sub_det in sub_dets:
						dets.append(sub_det)
				else:
					for index, (det, sub_det) in enumerate(zip(dets, sub_dets)):
						dets[index] = np.concatenate((det, sub_det), axis=0)

		sub_dets = inference_detector(model, img, template_img, 0.065)
		for index, (det, sub_det) in enumerate(zip(dets, sub_dets)):
			need_delete_index = []
			for temp_index, temp in enumerate(sub_det):
				if temp[2] - temp[0] < 48 or temp[3] - temp[1] < 48:
					need_delete_index.append(temp_index)

			if len(need_delete_index) > 0:
				sub_det = np.delete(sub_det, need_delete_index, axis=0)
			dets[index] = np.concatenate((det, sub_det), axis=0)
			if index == 12:
				need_delete_index = []
				temp_det = dets[index]
				for t_index, t in enumerate(temp_det):
					if t[2] - t[0] < width * 0.9 or t[3] - t[1] < height * 0.9:
						need_delete_index.append(t_index)

				dets[index] = np.delete(dets[index], need_delete_index, axis=0)

			dets[index] = nms(dets[index], 0.15)

		for index, bboxes in enumerate(dets):
			if index >= 15:
				continue
			bboxes = bboxes[0]
			for bbox in bboxes:
				if len(bbox) == 0:
					continue

				score = bbox[4].item()
				cls_type = get_cls_type(index)
				new_bbox = [round(bbox[0].item(), 2), round(bbox[1].item(), 2), round(bbox[2].item(), 2), round(bbox[3].item(), 2)]
				name = image_dir + '.jpg'
				result = {'name':name, 'category':cls_type, 'bbox':new_bbox, 'score':score}
				results.append(result)

		print("\r"+"{}/{}".format(image_index, count), end="", flush=True)

	print("use time = {}".format(time.time()-start))
	with open('../../submit/result.json', 'w') as fp:
		json.dump(results, fp, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	main()
