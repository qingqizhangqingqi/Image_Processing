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
		im = cv2.imread(image_path)
		return image_dir, im


	def __len__(self):
		return len(self.data_list)


def parse_args():
	parser = argparse.ArgumentParser(description='inference')
	parser.add_argument('--config', type=str, help='test config file path')
	parser.add_argument('--model', type=str, help='model file path')
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--thresh', type=float, default=0.5, help='bbox score threshold')
	args = parser.parse_args()
	return args

def get_cls_type(index):
	index_map = {'0':1, '1':2, '2':3, '3':4, '4':5, '5':6, '6':7, '7':8, '8':9, '9':10, '10':11, '11':12, '12':13, '13':14, '14':15}

	index = str(index)
	return index_map[index]

def main():
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = '6'
	model = init_detector(args.config, args.model, device=torch.device('cuda', args.gpu))
	model.set_socre_thr()

	root_dir = 'data/defect'
	#root_dir = "/tcdata/guangdong1_round2_testA_20190924"
	results = []

	dataset = MyDataset(root_dir)
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
	count = len(dataloader)
	start = time.time()
	for image_index, data in enumerate(dataloader):
		if image_index % 500 != 0:
			continue

		image_dir, img = data[0][0], data[1][0]
		img = img.numpy()
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
				sub_dets = inference_detector(model, sub_im)
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

		sub_dets = inference_detector(model, img)
		for index, (det, sub_det) in enumerate(zip(dets, sub_dets)):
			dets[index] = np.concatenate((det, sub_det), axis=0)
			dets[index] = nms(dets[index], 0.3)

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
				#cv2.rectangle(img, (int(new_bbox[0]), int(new_bbox[1])), (int(new_bbox[2]), int(new_bbox[3])), (255, 0, 0), 2)
				#cv2.putText(img, str(cls_type), (int(new_bbox[0])+10, int(new_bbox[1]+10)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0))

				#print("class type = {}, bbox = {}, score = {}".format(cls_type, bbox, score))
				name = image_dir + '.jpg'
				result = {'name':name, 'category':cls_type, 'bbox':new_bbox, 'score':score}
				results.append(result)

		print("\r"+"{}/{}".format(image_index, count), end="", flush=True)

	print("use time = {}".format(time.time()-start))
	with open('../result.json', 'w') as fp:
		json.dump(results, fp, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	main()
