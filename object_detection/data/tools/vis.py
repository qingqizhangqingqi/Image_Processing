import cv2
import os
import random
import xml.etree.ElementTree as ET

CLASSES = {'沾污':1, '错花':2, '水印':3, '花毛':4, '缝头':5, '缝头印':6, '虫粘':7, '破洞':8, '褶子':9,
        '织疵':10, '漏印':11, '蜡斑':12, '色差':13, '网折':14, '其他':15, '16':16}

def vis(image_dir, anno_dir, save_dir):
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	xml_files = os.listdir(anno_dir)
	total_len = len(xml_files)
	for index, xml_file in enumerate(xml_files):
		if random.randint(0, total_len-1) % 500 != 0:
		#if index % 500 != 0:
			continue
		base = xml_file.split('.')[0]
		if not os.path.exists(os.path.join(save_dir, base)):
			os.mkdir(os.path.join(save_dir, base))
		style = base.split('_')[0]
		image_file = os.path.join(base, base + '.jpg')
		template_file = os.path.join(base, "template_{}.jpg".format(style))
		template_path = os.path.join(image_dir, template_file)
		xml_path = os.path.join(anno_dir, xml_file)
		image_path = os.path.join(image_dir, image_file)
		image = cv2.imread(image_path)
		template_img = cv2.imread(template_path)
		(height, width, _) = image.shape
		(height, width, _) = template_img.shape
		#image = cv2.resize(image, (int(width/2), int(height/2)))
		#template_img = cv2.resize(template_img, (int(width/2), int(height/2)))

		root = ET.parse(xml_path)
		objs = root.findall('object')
		for obj in objs:
			cls = obj.find('name').text
			bndbox = obj.find('bndbox')
			#xmin = int(float(bndbox.find('xmin').text)/2)
			#ymin = int(float(bndbox.find('ymin').text)/2)
			#xmax = int(float(bndbox.find('xmax').text)/2)
			#ymax = int(float(bndbox.find('ymax').text)/2)
			xmin = int(float(bndbox.find('xmin').text))
			ymin = int(float(bndbox.find('ymin').text))
			xmax = int(float(bndbox.find('xmax').text))
			ymax = int(float(bndbox.find('ymax').text))
			cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
			cv2.rectangle(template_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
			cv2.putText(image, str(CLASSES[cls]), (xmin+30, ymin+30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255))
			cv2.putText(template_img, str(CLASSES[cls]), (xmin+30, ymin+30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255))

		image = cv2.resize(image, (int(width/2), int(height/2)))
		template_img = cv2.resize(template_img, (int(width/2), int(height/2)))
		save_file = os.path.join(save_dir, image_file)
		cv2.imwrite(save_file, image)
		save_template_file = os.path.join(save_dir, template_file)
		cv2.imwrite(save_template_file, template_img)
		print("{}/{}".format(index, total_len))

if __name__ == "__main__":
	vis('defect', 'Annotations', 'vis')
	#vis('new_defect', 'new_anno', 'vis')
	#vis('split_window', 'split_anno', 'vis')
	#vis('normal', 'normal_anno', 'vis')
