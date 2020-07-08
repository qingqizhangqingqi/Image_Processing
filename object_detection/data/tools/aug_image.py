import os
import json
import xml.etree.ElementTree as ET
import cv2
import random
import mmcv
import numpy as np

class_count = {'沾污':0, '错花':0, '水印':0, '花毛':0, '缝头':0, '缝头印':0, '虫粘':0, '破洞':0, '褶子':0,
        '织疵':0, '漏印':0, '蜡斑':0, '色差':0, '网折':0, '其他':0, '正常':0}

def indent(elem, level=0):
        i = "\n" + level*"\t"
        if len(elem):
                if not elem.text or not elem.text.strip():
                        elem.text = i + "\t"
                if not elem.tail or not elem.tail.strip():
                        elem.tail = i
                for elem in elem:
                        indent(elem, level+1)
                if not elem.tail or not elem.tail.strip():
                        elem.tail = i
        else:
                if level and (not elem.tail or not elem.tail.strip()):
                        elem.tail = i

def random_change(img, template):
	alpha = random.uniform(0.9,1.0)
	beta = random.randint(1, 8)
	blank = np.zeros(img.shape, img.dtype)
	new_img = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
	new_template = cv2.addWeighted(template, alpha, blank, 1-alpha, beta)
	return new_img, new_template


def translate(img, template, bboxes):
	h_img, w_img, _ = img.shape
	max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
	max_l_trans = max_bbox[0]
	max_u_trans = max_bbox[1]
	max_r_trans = w_img - max_bbox[2]
	max_d_trans = h_img - max_bbox[3]
 
	tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
	ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
 
	M = np.array([[1, 0, tx], [0, 1, ty]])
	img = cv2.warpAffine(img, M, (w_img, h_img))
	template = cv2.warpAffine(template, M, (w_img, h_img))
 
	bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
	bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
	return img, template, bboxes

def create_xml(anno_dir, image_dir, name, defect_names, bboxes, width, height):
	name_prefix = name.split('.')[0]
	xml_name = name_prefix + ".xml"
	xml_path = os.path.join(anno_dir, xml_name)

	root = ET.Element('annotation')
	xml_jpg_name = ET.SubElement(root, 'filename')
	xml_jpg_name.text = name
	xml_image_path = ET.SubElement(root, 'path')
	xml_image_path.text = os.path.join(image_dir, name_prefix, name)
	xml_size = ET.SubElement(root, 'size')
	xml_width = ET.SubElement(xml_size, 'width')
	xml_width.text = str(width)
	xml_height = ET.SubElement(xml_size, 'height')
	xml_height.text = str(height)
	xml_depth = ET.SubElement(xml_size, 'depth')
	xml_depth.text = '3'
	for defect_name, bbox in zip(defect_names, bboxes):
		xmin = bbox[0]
		ymin = bbox[1]
		xmax = bbox[2]
		ymax = bbox[3]

		xml_object = ET.SubElement(root, 'object')
		xml_object_diff = ET.SubElement(xml_object, 'difficult')
		xml_object_diff.text = '0'
		xml_object_name = ET.SubElement(xml_object, 'name')
		xml_object_name.text = defect_name
		xml_object_bndbox = ET.SubElement(xml_object, 'bndbox')
		xml_bndbox_xmin = ET.SubElement(xml_object_bndbox, 'xmin')
		xml_bndbox_xmin.text = str(xmin)
		xml_bndbox_ymin = ET.SubElement(xml_object_bndbox, 'ymin')
		xml_bndbox_ymin.text = str(ymin)
		xml_bndbox_xmax = ET.SubElement(xml_object_bndbox, 'xmax')
		xml_bndbox_xmax.text = str(xmax)
		xml_bndbox_ymax = ET.SubElement(xml_object_bndbox, 'ymax')
		xml_bndbox_ymax.text = str(ymax)


	indent(root)
	tree = ET.ElementTree(root)
	tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def parse_voc(anno_dir, min_size=8):
	anno_files = os.listdir(anno_dir)
	for i, anno_file in enumerate(anno_files):
		file_path = os.path.join(anno_dir, anno_file)
		root = ET.parse(file_path)
		size = root.find('size')
		objs = root.findall('object')
		gt_count = len(objs)
		for obj in objs:
			bbox = obj.find('bndbox')
			xmin = float(bbox.find('xmin').text)
			ymin = float(bbox.find('ymin').text)
			xmax = float(bbox.find('xmax').text)
			ymax = float(bbox.find('ymax').text)
			if xmax - xmin < min_size or ymax - ymin < min_size:
				continue

			name = obj.find('name').text
			class_count[name] = class_count[name] + 1

	for cls in class_count:
		print("key = {}, count = {}".format(cls, class_count[cls]))

def get_normal_list(style, normal_dir="normal"):
	path_list = []
	normal_files = os.listdir(normal_dir)
	for normal_file in normal_files:
		if normal_file.split('_')[0] == style:
			normal_path = os.path.join(normal_dir, normal_file, normal_file + '.jpg')
			path_list.append(normal_path)

	return path_list


def mix(defect_path, normal_path, objs, image_save_path, anno_save_path, index, random_change_flag=False):
	resize_ratio = 1.0
	base = defect_path.split('/')[1]
	new_filename = "{}{}".format(base, index)
	style = base.split('_')[0]
	defect_image_path = os.path.join(defect_path, defect_path.split('/')[1] + ".jpg")
	defect_temp_path = os.path.join(defect_path, "template_{}.jpg".format(style))
	normal_image_path = normal_path #os.path.join(normal_path, normal_path.split('/')[1] + '.jpg')
	defect_im = cv2.imread(defect_image_path)
	defect_temp = cv2.imread(defect_temp_path)	
	normal_im = cv2.imread(normal_image_path)
	height, width, _ = defect_im.shape
	height = int(height*resize_ratio)
	width = int(width*resize_ratio)
	defect_im = cv2.resize(defect_im, (width, height))
	defect_temp = cv2.resize(defect_temp, (width, height))
	normal_im = cv2.resize(normal_im, (width, height))

	defect_names = []
	bboxes = []
	for obj in objs:
		defect_name = obj.find('name').text
		bndbox = obj.find('bndbox')
		xmin = int(round(float(bndbox.find('xmin').text) * resize_ratio, 0))
		ymin = int(round(float(bndbox.find('ymin').text) * resize_ratio, 0))
		xmax = int(round(float(bndbox.find('xmax').text) * resize_ratio, 0))
		ymax = int(round(float(bndbox.find('ymax').text) * resize_ratio, 0))

		defect_names.append(defect_name)
		bboxes.append([xmin, ymin, xmax, ymax])
		xmin = int(xmin)
		if xmin > 0:
			xmin -= 1
		ymin = int(ymin)
		if ymin > 0:
			ymin -= 1
		xmax = int(xmax)
		if xmax < width:
			xmax += 1
		ymax = int(ymax)
		if ymax < height:
			ymax += 1
		normal_im[ymin:ymax, xmin:xmax, :] = defect_im[ymin:ymax, xmin:xmax, :]

	bboxes = np.array(bboxes)
	if len(defect_names) <= 0:
		return

	save_dir = os.path.join(image_save_path, new_filename)
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	im_save_path = os.path.join(save_dir, new_filename + ".jpg")
	temp_save_path = os.path.join(save_dir, "template_{}.jpg".format(style))
	#if random_change_flag:
	#	normal_im, defect_temp = random_change(normal_im, defect_temp)
	normal_im, defect_temp, bboxes = translate(normal_im, defect_temp, bboxes)
	
	cv2.imwrite(im_save_path, normal_im)
	cv2.imwrite(temp_save_path, defect_temp)
	create_xml(anno_save_path, image_save_path, new_filename, defect_names, bboxes, width, height)
	return


def aug_image(anno_dir, image_dir="defect", anno_save_dir="Annotations", image_save_dir="defect", gt_less=20, min_size=8):
	if not os.path.exists(anno_save_dir):
		os.mkdir(anno_save_dir)
	if not os.path.exists(image_save_dir):
		os.mkdir(image_save_dir)

	anno_files = os.listdir(anno_dir)
	count =len(anno_files)
	for index, anno_file in enumerate(anno_files):
		anno_path = os.path.join(anno_dir, anno_file)
		style = anno_file.split('_')[0]
		image_base = anno_file.split('.')[0]
		root = ET.parse(anno_path)
		objs = root.findall('object')
		gt_count = len(objs)
		image_file = os.path.join(image_dir, image_base)
		normal_list = get_normal_list(style)
		template_file = os.path.join(image_dir, image_base, "template_{}.jpg".format(style))
		normal_list.append(template_file)
		normal_count = len(normal_list)
		small_count = 0
		if gt_count > gt_less:
			normal_objs = []
			for obj in objs:
				defect_name = obj.find('name')
				bndbox = obj.find('bndbox')
				xmin = float(bndbox.find('xmin').text)
				ymin = float(bndbox.find('ymin').text)
				xmax = float(bndbox.find('xmax').text)
				ymax = float(bndbox.find('ymax').text)
				if xmax - xmin < min_size or ymax - ymin < min_size:
					continue

				if defect_name == '沾污' and random.random() < 0.75:
					continue

				if (defect_name == '错花' or defect_name == '水印') and random.random() < 0.5:
					continue

				normal_objs.append(obj)

			random.shuffle(normal_objs)
			gt_count = len(normal_objs)
			batch = int(gt_count/gt_less) + 1
			for i in range(batch):
				if i == batch - 1:
					obj_batch = normal_objs[i*gt_less:]
				else:
					obj_batch = normal_objs[i*gt_less:(i+1)*gt_less]

				normal_index = random.randint(0, normal_count-1)
				normal_file = normal_list[normal_index]
				new_index = index * 1000 + i
				mix(image_file, normal_file, obj_batch, image_save_dir, anno_save_dir, new_index)
		else:
			counts_thresh = [120, 200, 300, 500, 700]
			p_thresh = [1, 0.8, 0.6, 0.4, 0.2]
			for i, count_thresh in enumerate(counts_thresh):
				obj_batch = []
				for obj in objs:
					bndbox = obj.find('bndbox')
					xmin = float(bndbox.find('xmin').text)
					ymin = float(bndbox.find('ymin').text)
					xmax = float(bndbox.find('xmax').text)
					ymax = float(bndbox.find('ymax').text)
					if xmax - xmin < min_size or ymax - ymin < min_size:
						continue

					defect_name = obj.find('name').text
					if class_count[defect_name] < count_thresh and random.random() < p_thresh[i]:
						obj_batch.append(obj)

				if len(obj_batch) <= 0:
					continue

				normal_index = random.randint(0, normal_count-1)
				normal_file = normal_list[normal_index]
				new_index = index * 1000 + i
				mix(image_file, normal_file, obj_batch, image_save_dir, anno_save_dir, new_index, False)

		print("\r"+"{}/{}".format(index, count), end="", flush=True)

	
if __name__ == "__main__":
	min_size = 8
	parse_voc("Annotations", min_size=min_size)
	aug_image("Annotations", min_size=min_size)
