import argparse
import os
import cv2
import torch
from mmdet.apis import inference_detector, init_detector, show_result
import json
import numpy as np
from mmdet.ops.nms.nms_wrapper import nms

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
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	model = init_detector(args.config, args.model, device=torch.device('cuda', args.gpu))

	root_dir = "data/normal"
	#root_dir = "/tcdata/guangdong1_round2_testA_20190924"
	#root_dir = "/data/guangdong1_round2_testA_20190924"
	image_dirs = os.listdir(root_dir)
	count = len(image_dirs)
	results = []

	for image_index, image_dir in enumerate(image_dirs):
		if not os.path.isdir(os.path.join(root_dir, image_dir)):
			continue
		image_files = os.listdir(os.path.join(root_dir, image_dir))
		image_path = os.path.join(root_dir, image_dir, image_dir + '.jpg')
		img = cv2.imread(image_path)
		style = image_dir.split('_')[0]
		template_path = os.path.join(root_dir, image_dir, "template_{}.jpg".format(style))
		template_img = cv2.imread(template_path)
		result = inference_detector(model, img, template_img)
		for index, bboxes in enumerate(result):
			if index >= 15:
				continue
			for bbox in bboxes:
				score = bbox[4].item()
				cls_type = get_cls_type(index)
				new_bbox = [round(bbox[0].item(), 2), round(bbox[1].item(), 2), round(bbox[2].item(), 2), round(bbox[3].item(), 2)]
				#print("class type = {}, bbox = {}, score = {}".format(cls_type, bbox, score))
				name = image_dir + '.jpg'
				result = {'name':name, 'category':cls_type, 'bbox':new_bbox, 'score':score}
				results.append(result)

		print("\r"+"{}/{}".format(image_index, count), end="", flush=True)
	with open('../result.json', 'w') as fp:
		json.dump(results, fp, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	main()
