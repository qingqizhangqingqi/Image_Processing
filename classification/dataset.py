import os
import cv2
import numpy.random as random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import csv

def random_flip(img):
	if random.random() < 0.5:
		return img
	return cv2.flip(img, 1)

def translate(img):
	height, width, _ = img.shape
	x0 = int(time.time() % 10)
	y0 = int(time.time() % 10)
	x1 = int(width - random.randint(0, 10))
	y1 = int(height - random.random() * 10)
	img = img[y0:y1, x0:x1, :]
	return img
	
def resize(img, width, height):
	return cv2.resize(img, (width, height))

def contrast_brightness(img, min_alpha=0.75, max_beta=25):
	if random.random() < 0.5:
		return img

	blank = np.zeros(img.shape, img.dtype)
	alpha = random.uniform(min_alpha, 1.0)
	beta = random.randint(0, max_beta)
	return cv2.addWeighted(img, alpha, blank, 1-alpha, beta)

def GaussianBlur(img, kernels=[3, 5]):
	if random.random() < 0.5:
		return img
	kernel_index = random.randint(0, len(kernels))
	kernel_size = kernels[kernel_index]
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def convert_color_factory(src, dst):
    code = getattr(cv2, 'COLOR_{}2{}'.format(src.upper(), dst.upper()))
    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = """Convert a {0} image to {1} image.
    Args:
        img (ndarray or str): The input image.
    Returns:
        ndarray: The converted {1} image.
    """.format(src.upper(), dst.upper())

    return convert_color

bgr2rgb = convert_color_factory('bgr', 'rgb')
def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    if to_rgb:
        img = bgr2rgb(img)
    return (img - mean) / std

def reset_image(img, width, height, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
	img = translate(img)
	img = resize(img, width, height)
	img = random_flip(img)
	img = contrast_brightness(img, 0.95, 5)
	img = GaussianBlur(img)
	img = imnormalize(img, mean, std)
	return img

class TestDataset(Dataset):
	def __init__(self, data_dir, anno_file, width, height):
		self.data_dir = data_dir
		self.width = width
		self.height = height
		self.file_list = []

		fp = open(anno_file)
		reader = csv.reader(fp)
		for index, row in enumerate(reader):
			if index == 0:
				continue
			self.file_list.append(row)
		fp.close()

	def __getitem__(self, index):
		file_info = self.file_list[index]
		image_file = file_info[0]
		image_path = os.path.join(self.data_dir, image_file + '.jpg')
		im = cv2.imread(image_path)
		im = resize(im, self.width, self.height)
		im = imnormalize(im, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
		im = np.array(im, dtype=np.float32)
		if len(file_info) == 2:
			category = file_info[1]
			return im, int(category)
		else:
			return im

	def __len__(self):
		return len(self.file_list)

class ImageDataset(Dataset):
	def __init__(self, data_dir, anno_file, width, height):
		self.data_dir = data_dir
		self.width = width
		self.height = height
		self.file_list = []
		fp = open(anno_file)
		reader = csv.reader(fp)
		for index, row in enumerate(reader):
			if index == 0:
				continue
			self.file_list.append(row)
		fp.close()

	def __getitem__(self, index):
		file_info = self.file_list[index]
		image_file = file_info[0]
		category = file_info[1]
		image_path = os.path.join(self.data_dir, image_file + '.jpg')
		im = reset_image(cv2.imread(image_path), self.width, self.height)
		im = np.array(im, dtype=np.float32)
		return im, int(category)

	def __len__(self):
		return len(self.file_list)
