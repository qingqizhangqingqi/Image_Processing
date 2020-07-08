import argparse
import os
import cv2
import torch
from mmdet.apis import inference_detector, init_detector
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
		style = image_dir.split('_')[0]
		template_path = os.path.join(self.data_dir, image_dir, 'template_{}.jpg'.format(style))
		im = cv2.imread(image_path)
		template = cv2.imread(template_path)
		return image_dir, im, template


	def __len__(self):
		return len(self.data_list)


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

def main(anno_save_dir):
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	model = init_detector(args.config, args.model, device=torch.device('cuda', args.gpu))

	root_dir = "../../data/normal"
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
				sub_dets = inference_detector(model, sub_im, sub_template)
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

		sub_dets = inference_detector(model, img, template_img)
		for index, (det, sub_det) in enumerate(zip(dets, sub_dets)):
			dets[index] = np.concatenate((det, sub_det), axis=0)
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
				if new_bbox[2] - new_bbox[0] < 10 or new_bbox[3] - new_bbox[1] < 8:
					continue
				create_xml(anno_save_dir, root_dir, image_dir+'.jpg', "16", new_bbox)

		print("\r"+"{}/{}".format(image_index, count), end="", flush=True)

	print("use time = {}".format(time.time()-start))

if __name__ == '__main__':
	anno_save_dir = "../../data/normal_anno"
	if not os.path.exists(anno_save_dir):
		os.mkdir(anno_save_dir)
	
	main(anno_save_dir)
