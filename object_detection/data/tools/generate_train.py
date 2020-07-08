import os
import xml.etree.ElementTree as ET
import random

def main(anno_dir, image_dir):
	gt_less = 20
	val_list = []
	train_list = []
	trainval_list= []
	anno_files = os.listdir(anno_dir)
	if not os.path.exists('ImageSets'):
		os.mkdir('ImageSets')
	if not os.path.exists('ImageSets/Main'):
		os.mkdir('ImageSets/Main')

	for i, anno_file in enumerate(anno_files):
		image_base = anno_file.split('.')[0] + '\n'
		anno_path = os.path.join(anno_dir, anno_file)
		root = ET.parse(anno_path)
		objs = root.findall('object')
		if len(objs) > gt_less:
			val_list.append(image_base)
		else:
			if i % 15 == 0:
				val_list.append(image_base)
			else:
				train_list.append(image_base)
			trainval_list.append(image_base)

	random.shuffle(val_list)
	random.shuffle(train_list)
	random.shuffle(trainval_list)
	val_f = open('ImageSets/Main/val.txt', 'w')
	for val in val_list:
		val_f.write(val)
	val_f.close()
	train_f = open('ImageSets/Main/train.txt', 'w')
	for train in train_list:
		train_f.write(train)
	train_f.close()

	trainval_f = open('ImageSets/Main/trainval.txt', 'w')
	for line in trainval_list:
		trainval_f.write(line)
	trainval_f.close()

if __name__ == "__main__":
	main('Annotations', 'defect')


