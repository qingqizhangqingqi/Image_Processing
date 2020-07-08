import os
import json
import xml.etree.ElementTree as ET
import cv2
import sys
import math

WIDTH = 4096
HEIGHT = 1696
class_count = {'沾污':0, '错花':0, '水印':0, '花毛':0, '缝头':0, '缝头印':0, '虫粘':0, '破洞':0, '褶子':0,
        '织疵':0, '漏印':0, '蜡斑':0, '色差':0, '网折':0, '其他':0, '16':0}

small_count = {'沾污':0, '错花':0, '水印':0, '花毛':0, '缝头':0, '缝头印':0, '虫粘':0, '破洞':0, '褶子':0,
        '织疵':0, '漏印':0, '蜡斑':0, '色差':0, '网折':0, '其他':0, '16':0}

def set_scale_list(scale_list, scale, scales):
	count = len(scale_list)
	for i in range(count):
		s = scales[i]
		if scale <= s:
			scale_list[i] += 1
			break

def parse_voc(train_file):
	max_scale = 0
	min_scale = 10000
	min_width = 10000
	min_height = 10000
	#0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92, 100
	base = 0.01
	scale_list = [0, 0, 0, 0, 0, 0 ,0 ,0, 0, 0, 0, 0, 0]
	scales = [0.04, 0.32, 0.64, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 54, 80.0, 1000.0, 10000.0]
	tmp = 0
	gt_less = 50

	f = open(train_file)
	lines = f.read().splitlines()
	for i, line in enumerate(lines):
		file_path = os.path.join("Annotations", '{}.xml'.format(line))
		root = ET.parse(file_path)

		size = root.find('size')
		image_width = int(size.find('width').text)
		image_height = int(size.find('height').text)
		objs = root.findall('object')
		gt_count = len(objs)
		if gt_count > gt_less:
			tmp += 1
			continue

		for obj in objs:
			name = obj.find('name').text
			size = obj.find('bndbox')
			xmin = float(size.find('xmin').text)
			ymin = float(size.find('ymin').text)
			xmax = float(size.find('xmax').text)
			ymax = float(size.find('ymax').text)
			
			width = xmax - xmin
			height = ymax - ymin
			if width < 8 or height < 8:
				small_count[name] += 1

			scale = width / height
			set_scale_list(scale_list, scale, scales)
			if scale < min_scale:
				min_scale = scale
			if scale > max_scale:
				max_scale = scale

		
			class_count[name] = class_count[name] + 1

	for cls in class_count:
		print("key = {}, count = {}, small count = {}".format(cls, class_count[cls], small_count[cls]))
	
	print("gt > {}, total count = {}/{}".format(gt_less, tmp, len(lines)))
	print("max scale = {}, min scale = {}, min width = {}, min height = {}".format(max_scale, min_scale, min_width, min_height))
	print(scales)
	print(scale_list)

if __name__ == "__main__":
	parse_voc("ImageSets/Main/trainval.txt")

