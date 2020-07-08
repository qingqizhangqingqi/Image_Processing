import os
import json
import xml.etree.ElementTree as ET
import cv2
import random

class_count = {'沾污':0, '错花':0, '水印':0, '花毛':0, '缝头':0, '缝头印':0, '虫粘':0, '破洞':0, '褶子':0,
        '织疵':0, '漏印':0, '蜡斑':0, '色差':0, '网折':0, '其他':0}

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

def create_dir():
	annotation_dir = "Annotations"
	if not os.path.exists(annotation_dir):
		os.mkdir(annotation_dir)

	imageset_dir = "ImageSets"
	if not os.path.exists(imageset_dir):
		os.mkdir(imageset_dir)

	main_dir = os.path.join(imageset_dir, "Main")
	if not os.path.exists(main_dir):
		os.mkdir(main_dir)

	return annotation_dir, main_dir, "defect"
	
def create_xml(anno_dir, image_dir, name, defect_name, bbox):
	name_prefix = name.split('.')[0]
	xml_name = name_prefix + ".xml"
	xml_path = os.path.join(anno_dir, xml_name)
	image_file = os.path.join(image_dir, name_prefix, name)
	xmin = int(bbox[0])
	ymin = int(bbox[1])
	xmax = int(bbox[2])
	ymax = int(bbox[3])
	
	if os.path.exists(xml_path):
		tree = ET.parse(xml_path)
		root = tree.getroot()
		xml_object = ET.SubElement(root, 'object')
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
		tree.write(xml_path, encoding="utf-8", xml_declaration=True)
	else:
		im = cv2.imread(image_file)
		width = im.shape[1]
		height = im.shape[0]
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

		xml_object = ET.SubElement(root, 'object')
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


def parse_json_file(annotation_dir, imageset_dir, image_dir, file_path):
	train_file_path = os.path.join(imageset_dir, "train.txt")
	val_file_path = os.path.join(imageset_dir, "val.txt")
	train_file = open(train_file_path, "w")
	val_file = open(val_file_path, "w")

	file = open(file_path, "rb")
	file_json = json.load(file)
	for i, item in enumerate(file_json):
		name = item['name']
		if '917A1' in name: 
			print("problem picture, name = {}".format(name))
			continue

		defect_name = item['defect_name']
		bbox = item['bbox']
		print("i = {}, name = {}, defect_name = {}, bbox = {}".format(i, name, defect_name, bbox))
		create_xml(annotation_dir, image_dir, name, defect_name, bbox)

	gt_threshold = 20
	val_list = []
	train_list = []
	anno_files = os.listdir(annotation_dir)
	for index, anno_file in enumerate(anno_files):
		file_prefix = anno_file.split('.')[0]
		anno_path = os.path.join(annotation_dir, anno_file)
		root = ET.parse(anno_path)
		objs = root.findall('object')
		val_list.append(file_prefix)
		continue

		if len(objs) > gt_threshold:
			val_list.append(file_prefix)
		else:
			if index % 15 == 0:
				val_list.append(file_prefix)
			else:
				train_list.append(file_prefix)
		#file_list = val_list if index % 10 == 0 else train_list
		#file_list.append(file_prefix)

	random.shuffle(val_list)
	random.shuffle(train_list)
	for val in val_list:
		val_file.write(val + '\n')

	for train in train_list:
		train_file.write(train + '\n')

	train_file.close()
	val_file.close()


if __name__ == "__main__":
	annotation_dir, imageset_dir, image_dir = create_dir()
	anno_files = os.listdir('json_anno')
	for anno_file in anno_files:
		anno_path = os.path.join('json_anno', anno_file)
		parse_json_file(annotation_dir, imageset_dir, image_dir, anno_path)

