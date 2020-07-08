import cv2
import os
from skimage.measure import compare_ssim
import numpy as np
import xml.etree.ElementTree as ET

def to_gray(root_dir, save_dir, show_bbox=False):
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	image_dirs = os.listdir(root_dir)
	count = len(image_dirs)
	for index, image_dir in enumerate(image_dirs):
		image_full_dir = os.path.join(root_dir, image_dir)
		if not os.path.isdir(image_full_dir):
			continue

		full_save_dir = 'JPEGImages'
		if show_bbox:
			full_save_dir = os.path.join(save_dir, image_dir)

		if not os.path.exists(full_save_dir):
			os.mkdir(full_save_dir)

		style = image_dir.split('_')[0]
		image_path = os.path.join(image_full_dir, image_dir + '.jpg')
		template_path = os.path.join(image_full_dir, "template_{}.jpg".format(style))
		im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		height, width = im.shape
		template_im = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
		if show_bbox:
			im = cv2.resize(im, (int(width/2), int(height/2)))
			template_im = cv2.resize(template_im, (int(width/2), int(height/2)))

			im_source_a = cv2.imread(image_path)
			im_source_b = cv2.imread(template_path)
			im_source_a = cv2.resize(im_source_a, (int(width/2), int(height/2)))
			im_source_b = cv2.resize(im_source_b, (int(width/2), int(height/2)))

	
			im_save_path = os.path.join(full_save_dir, image_dir + '_gray.jpg')
			template_save_path = os.path.join(full_save_dir, "template_{}_gray.jpg".format(style))


		#(score, diff) = compare_ssim(im, template_im, full=True)
		#diff = (diff * 255).astype('uint8')
		diff = (im - template_im).astype('uint8')
		im_a = np.expand_dims(im, axis=2)
		im_b = np.expand_dims(template_im, axis=2)
		im_c = np.expand_dims(diff, axis=2)
		new_image = np.concatenate((im_a, im_b, im_c), axis=2)
		new_save_path = os.path.join(full_save_dir, "{}.jpg".format(image_dir))

		if show_bbox:
			xml_file = os.path.join('Annotations', image_dir + '.xml')
			root = ET.parse(xml_file)
			objs = root.findall('object')
			for obj in objs:
				bndbox = obj.find('bndbox')
				xmin = int(bndbox.find('xmin').text)
				ymin = int(bndbox.find('ymin').text)
				xmax = int(bndbox.find('xmax').text)
				ymax = int(bndbox.find('ymax').text)
				cv2.rectangle(im, (int(xmin/2), int(ymin/2)), (int(xmax/2), int(ymax/2)), (255, 0, 0))
				cv2.rectangle(template_im, (int(xmin/2), int(ymin/2)), (int(xmax/2), int(ymax/2)), (255, 0, 0))
				cv2.rectangle(diff, (int(xmin/2), int(ymin/2)), (int(xmax/2), int(ymax/2)), (255, 0, 0))
				cv2.rectangle(new_image, (int(xmin/2), int(ymin/2)), (int(xmax/2), int(ymax/2)), (255, 0, 0))
				cv2.rectangle(im_source_a, (int(xmin/2), int(ymin/2)), (int(xmax/2), int(ymax/2)), (255, 0, 0))
				cv2.rectangle(im_source_b, (int(xmin/2), int(ymin/2)), (int(xmax/2), int(ymax/2)), (255, 0, 0))

			diff_save_path = os.path.join(full_save_dir, "{}_diff.jpg".format(image_dir))
			cv2.imwrite(diff_save_path, diff)
			cv2.imwrite(im_save_path, im)
			cv2.imwrite(template_save_path, template_im)
			source_a_save = os.path.join(full_save_dir, "{}_source_d.jpg".format(image_dir))
			source_b_save = os.path.join(full_save_dir, "{}_source_t.jpg".format(image_dir))
			cv2.imwrite(source_a_save, im_source_a)
			cv2.imwrite(source_b_save, im_source_b)

		cv2.imwrite(new_save_path, new_image)
		print("{}/{}".format(index, count))

if __name__ == "__main__":
	to_gray('defect', 'gray_defect', False)
	#to_gray('normal', 'gray_normal')
	
