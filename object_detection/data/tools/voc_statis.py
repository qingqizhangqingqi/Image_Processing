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

def set_scale_list(scale_list, scale, scales):
	count = len(scale_list)
	for i in range(count):
		s = scales[i]
		if scale <= s:
			scale_list[i] += 1
			break

def parse_voc(file_dir):
	max_scale = 0
	min_scale = 10000
	min_width = 10000
	min_height = 10000
	max_area = 0
	#0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92, 100
	base = 0.01
	scale_list = [0, 0, 0, 0, 0, 0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0]
	scales = [0.04, 0.32, 0.64, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 54, 80.0, 1000.0, 10000.0]
	anno_files = os.listdir(file_dir)
	tmp = 0
	gt_less = 1000
	area_less_than = 64
	area_less_count = 0
	total_count = 0
	for i, anno_file in enumerate(anno_files):
		file_path = os.path.join(file_dir, anno_file)
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
			if width <= 0 or height <= 0:
				#print("xmin = {}, xmax = {}, ymin = {}, ymax = {}, file = {}".format(xmin, xmax, ymin, ymax, anno_file))
				continue

			if width < 32:
				#print("min width = {}, height = {}, name = {}, file = {}".format(min_width, height, name, anno_file))
				min_width = width

			if height < 32:
				#print("min height = {}, width = {},  name = {}, file = {}".format(min_height, width, name, anno_file))
				min_height = height

			area = (width) * (height)
			if area > max_area:
				max_area = area
				#print("xmin = {}, xmax = {}, ymin = {}, ymax = {}, name = {}".format(xmin, xmax, ymin, ymax, name))
			if area < area_less_than:
				#print("xmin = {}, xmax = {}, ymin = {}, ymax = {}, area = {}, name = {}, file = {}".format(xmin, xmax, ymin, ymax, area, name, anno_file))
				area_less_count += 1

			scale = width / height
			set_scale_list(scale_list, scale, scales)
			if scale < min_scale:
				min_scale = scale
			if scale > max_scale:
				max_scale = scale

		
			class_count[name] = class_count[name] + 1
			total_count += 1

	for cls in class_count:
		print("key = {}, count = {}".format(cls, class_count[cls]))
	
	print("gt > {}, total count = {}/{}".format(gt_less, tmp, len(anno_files)))
	print("max scale = {}, min scale = {}, min width = {}, min height = {}".format(max_scale, min_scale, min_width, min_height))
	print("max area = {}, area less count = {}/{}".format(max_area, area_less_count, total_count))
	print(scales)
	print(scale_list)

if __name__ == "__main__":
	parse_voc("Annotations")
	#parse_voc("new_anno")
