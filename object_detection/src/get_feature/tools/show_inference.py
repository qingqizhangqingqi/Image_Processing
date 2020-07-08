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
import xml.etree.ElementTree as ET

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

	show_source = False
	vis_dir = 'vis'
	if not os.path.exists(vis_dir):
		os.mkdir(vis_dir)

	anno_dir = 'data/Annotations'
	root_dir = 'data/normal'
	#root_dir = "/tcdata/guangdong1_round2_testA_20190924"
	results = []

	#dataset = MyDataset(root_dir)
	#dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
	#count = len(dataloader)
	image_dirs = os.listdir(root_dir)
	count = len(image_dirs)
	start = time.time()

	#for image_index, data in enumerate(dataloader):
	for image_index, image_dir in enumerate(image_dirs):
		if not os.path.isdir(os.path.join(root_dir, image_dir)):
			continue
		if image_index % 10 != 0:
			continue

		result_dir = os.path.join(vis_dir, "{}".format(image_index))
		if not os.path.exists(result_dir):
			os.mkdir(result_dir)
		#image_dir, img = data[0][0], data[1][0]
		#img = img.numpy()
		image_files = os.listdir(os.path.join(root_dir, image_dir))
		image_path = os.path.join(root_dir, image_dir, image_dir + '.jpg')
		img = cv2.imread(image_path)
		if show_source:
			source_path = os.path.join(root_dir, image_dir, image_dir + '.jpg')
			source_im = cv2.imread(source_path)
			style = image_dir.split('_')[0]
			template_path = os.path.join(root_dir, image_dir, "template_{}.jpg".format(style))
			template_im = cv2.imread(template_path)
			anno_path = os.path.join(anno_dir, image_dir + '.xml')
			if not os.path.exists(anno_path):
				continue

			root = ET.parse(anno_path)
			objs = root.findall('object')
			for obj in objs:
				cls = obj.find('name').text
				bndbox = obj.find('bndbox')
				xmin = int(float(bndbox.find('xmin').text))
				ymin = int(float(bndbox.find('ymin').text))
				xmax = int(float(bndbox.find('xmax').text))
				ymax = int(float(bndbox.find('ymax').text))
				#print("cls = {}, xmin = {}, xmax = {}, ymin = {}, ymax = {}".format(cls, xmin, xmax, ymin, ymax))
				cv2.rectangle(source_im, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
				cv2.putText(source_im, cls, (xmin+40, ymin+40), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
				cv2.rectangle(template_im, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
				cv2.putText(template_im, cls, (xmin+40, ymin+40), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))

			cv2.imwrite("{}/{}_source.jpg".format(result_dir, image_index), source_im)
			cv2.imwrite("{}/{}_template.jpg".format(result_dir, image_index), template_im)

		iter_start = time.time()
		height, width, _ = img.shape
		sub_height, sub_width = int(height/2), int(width/2)
		dets = []
		for row in range(2):
			for col in range(2):
				height_start = row * sub_height
				height_end = (row+1) * sub_height
				width_start = col * sub_width
				width_end = (col+1) * sub_width
				sub_im = img[height_start:height_end, width_start:width_end, :].copy()
				sub_dets = inference_detector(model, sub_im, 0.15)

				message = ""
				for index, sub_det in enumerate(sub_dets):
					for temp_box in sub_det:
						score = temp_box[4]
						show_message = "{}:{}".format(index+1, score)
						cv2.rectangle(sub_im, (int(temp_box[0]), int(temp_box[1])), (int(temp_box[2]), int(temp_box[3])), (255, 0, 0), 2)
						cv2.putText(sub_im, show_message, (int(temp_box[0])+40, int(temp_box[1]+40)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
					message += ", {}:{}".format(index, sub_det.shape[0])
					

				cv2.imwrite("{}/{}_{}_{}.jpg".format(result_dir, image_index, row, col), sub_im)

				print("row:{}, col:{}{}".format(row, col, message))
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


		temp_img = img.copy()
		sub_dets = inference_detector(model, img, 0.05)
		message1 = ""
		message2 = ""
		message3 = ""
		for index, (det, sub_det) in enumerate(zip(dets, sub_dets)):
			for temp_box in sub_det:
				score = temp_box[4]
				show_message = "{}:{}".format(index+1, score)
				cv2.rectangle(temp_img, (int(temp_box[0]), int(temp_box[1])), (int(temp_box[2]), int(temp_box[3])), (255, 0, 0), 2)
				cv2.putText(temp_img, show_message, (int(temp_box[0])+40, int(temp_box[1]+40)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))

			message1 += ", {}:{}".format(index, sub_det.shape[0])
			dets[index] = np.concatenate((det, sub_det), axis=0)
			message2 += ", {}:{}".format(index, dets[index].shape[0])
			dets[index] = nms(dets[index], 0.01)
			message3 += ", {}:{}".format(index, dets[index][0].shape[0])
		cv2.imwrite("{}/{}_full_image.jpg".format(result_dir, image_index), temp_img)
		print("image      ", message1, '\n', "cat       ", message2, '\n', "nms       ", message3)

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
				cv2.rectangle(img, (int(new_bbox[0]), int(new_bbox[1])), (int(new_bbox[2]), int(new_bbox[3])), (255, 0, 0), 2)
				message = "{}:{}".format(cls_type, score)
				cv2.putText(img, message, (int(new_bbox[0])+40, int(new_bbox[1]+40)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))

				#print("class type = {}, bbox = {}, score = {}".format(cls_type, bbox, score))
				name = image_dir + '.jpg'
				result = {'name':name, 'category':cls_type, 'bbox':new_bbox, 'score':score}
				results.append(result)


		iter_end = time.time()
		cv2.imwrite("{}/{}.jpg".format(result_dir, image_index), img)

		print("{}/{}, use time = {}".format(image_index, count, iter_end-iter_start))
		#print("\r"+"{}/{}, use time = {}".format(image_index, count, iter_end-iter_start), end="", flush=True)

	print("total use time = {}".format(time.time()-start))
	with open('../result.json', 'w') as fp:
		json.dump(results, fp, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	main()
