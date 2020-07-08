import argparse
import os
import cv2
import torch
from mmdet.apis import inference_detector, init_detector, show_result
import json
import mmcv
import numpy as np
from mmdet.ops.nms.nms_wrapper import nms
from torch.utils.data import Dataset, DataLoader
import time

class MyDataset(Dataset):
        def __init__(self, val_path, data_dir):
                val_f = open(val_path)
                lines = val_f.read().splitlines()
                self.data_list = []
                for data in lines:
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
	parser.add_argument('--run_gpu', type=str, default='0', help='run gpu')
	args = parser.parse_args()
	return args

def get_cls_type(index):
	index_map = {'0':1, '1':2, '2':3, '3':4, '4':5, '5':6, '6':7, '7':8, '8':9, '9':10, '10':11, '11':12, '12':13, '13':14, '14':15}

	index = str(index)
	return index_map[index]

def main():
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.run_gpu
	model = init_detector(args.config, args.model, device=torch.device('cuda', args.gpu))
	results = []

	dataset = MyDataset('../../data/ImageSets/Main/val.txt', '../../data/defect')
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
	count = len(dataloader)
	start = time.time()
	for image_index, data in enumerate(dataloader):
		iter_start = time.time()
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
				sub_dets = inference_detector(model, sub_im, sub_template, 0.1)
				for index, sub_det in enumerate(sub_dets):
					for sub_item in sub_det:
						sub_item[0] += width_start
						sub_item[1] += height_start
						sub_item[2] += width_start
						sub_item[3] += height_start

				if len(dets) == 0:
					for sub_det in sub_dets:
						dets.append(sub_det)
				else:
					for index, (det, sub_det) in enumerate(zip(dets, sub_dets)):
						dets[index] = np.concatenate((det, sub_det), axis=0)

		sub_dets = inference_detector(model, img, template_img, 0.1)
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
					if t[2] - t[0] < width * 0.9 or t[3] - t[1] <height * 0.9:
						need_delete_index.append(t_index)

				dets[index] = np.delete(dets[index], need_delete_index, axis=0)

			dets[index] = nms(dets[index], 0.15)[0]

		results.append(dets)
		iter_end = time.time()
		print("\r"+"{}/{}, use time = {}".format(image_index, count, iter_end-iter_start), end="", flush=True)

	mmcv.dump(results, 'eval/result.pkl')

if __name__ == '__main__':
	main()
