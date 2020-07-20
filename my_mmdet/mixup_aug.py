import math
import mmcv
import cv2
import numpy as np
from numpy import random

def bbox_flip(bboxes, img_shape, direction):
	assert bboxes.shape[-1] % 4 == 0
	flipped = bboxes.copy()
	if direction == 'horizontal':
		w = img_shape[1]
		flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
		flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
	elif direction == 'vertical':
		h = img_shape[0]
		flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
		flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
	else:
		raise ValueError('Invalid flipping direction "{}"'.format(direction))
	return flipped

def random_flip(img, bboxes, ratio=0.5):
	if random.random() > ratio or len(bboxes) == 0:
		return img, bboxes

	img = mmcv.imflip(img, direction='horizontal')
	bboxes = bbox_flip(bboxes, img.shape, 'horizontal')
	return img, bboxes

def gaussian_blur(img, kernels=[3, 5, 7, 9], ratio=0.5):
	if random.random() > ratio:
		return img

	kernel_index = random.randint(0, len(kernels))
	kernel_size = kernels[kernel_index]
	img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
	return img

def motion_blur(img, min_degree=1, max_degree=6, min_angle=30, max_angle=60, ratio=0.5):
	if random.random() > ratio:
		return img

	degree = random.randint(min_degree, max_degree)
	angle = random.randint(min_angle, max_angle)

	img = np.array(img)
	M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
	motion_blur_kernel = np.diag(np.ones(degree))
	motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

	motion_blur_kernel = motion_blur_kernel / degree
	blurred = cv2.filter2D(img, -1, motion_blur_kernel)
	
	cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
	blurred = np.array(blurred, dtype=np.uint8)
	return blurred

def random_vflip(img, bboxes, ratio=0.5):
	if random.random() > ratio or len(bboxes) == 0:
		return img, bboxes

	img = mmcv.imflip(img, direction="vertical")
	bboxes = bbox_flip(bboxes, img.shape, 'vertical')
	return img, bboxes

def rotate(img, bboxes, angle=90, ratio=0.5):
	if random.random() > ratio or len(bboxes) == 0:
		return img, bboxes

	scale = 1.0
	w = img.shape[1]
	h = img.shape[0]
	rangle = np.deg2rad(angle) # angle in radians
	# now calculate new image width and height
	nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
	nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
	# ask OpenCV for the rotation matrix
	rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
	# calculate the move from the old center to the new center combined
	# with the rotation
	rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
	# the move only affects the translation, so update the translation
	# part of the transform
	rot_mat[0,2] += rot_move[0]
	rot_mat[1,2] += rot_move[1]
	# 仿射变换
	rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

	rot_bboxes = list()
	for bbox in bboxes:
		xmin = bbox[0]
		ymin = bbox[1]
		xmax = bbox[2]
		ymax = bbox[3]
		point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
		point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
		point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
		point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
		
		concat = np.vstack((point1, point2, point3, point4))
		concat = concat.astype(np.int32)
		rx, ry, rw, rh = cv2.boundingRect(concat)
		rot_bboxes.append([rx, ry, rx+rw, ry+rh])

	rot_bboxes = np.array(rot_bboxes).astype('float32')
	return rot_img, rot_bboxes

def contrast_brightness(img, min_alpha=0.75, max_beta=25, ratio=0.5):
	if random.random() > ratio:
		return img

	blank = np.zeros(img.shape, img.dtype)
	alpha = random.uniform(min_alpha, 1.0)
	beta = random.randint(0, max_beta)
	img = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
	return img

def translate(img, bboxes, ratio=0.5):
	if random.random() > ratio or len(bboxes) == 0:
		return img, bboxes


	h_img, w_img, _ = img.shape
	min_x = min_y = 9999
	max_x = max_y = 0
	min_x = min_x if min_x < np.min(bboxes[:, 0], axis=0) else np.min(bboxes[:, 0], axis=0)
	min_y = min_y if min_y < np.min(bboxes[:, 1], axis=0) else np.min(bboxes[:, 1], axis=0)            
	max_x = max_x if max_x > np.max(bboxes[:, 2], axis=0) else np.max(bboxes[:, 2], axis=0)
	max_y = max_y if max_y > np.max(bboxes[:, 3], axis=0) else np.max(bboxes[:, 3], axis=0)            

	max_l_trans = min_x
	max_u_trans = min_y
	max_r_trans = w_img - max_x
	max_d_trans = h_img - max_y
 
	tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
	ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
 
	M = np.array([[1, 0, tx], [0, 1, ty]])
	img = cv2.warpAffine(img, M, (w_img, h_img))

	bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
	bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
	return img, bboxes

def grid_mask(img, bboxes, ratio=0.5):
	if random.random() > ratio or len(bboxes) <= 0:
		return img

	obj_ws = bboxes[:, 2] - bboxes[:, 0]
	obj_hs = bboxes[:, 3] - bboxes[:, 1]
	if len(obj_ws) == 0 or len(obj_hs) == 0:
		return img

	w_min = int(obj_ws.min())
	h_min = int(obj_hs.min())

	x = (w_min // 2) if (w_min // 2) <= (w_min // 1.5) else random.randint(w_min // 2, w_min // 1.5)
	y = (h_min // 2) if (h_min // 2) <= (h_min // 1.5) else random.randint(h_min // 2, h_min // 1.5)
	r_x = (w_min // 1) if (w_min // 1.5) <= (w_min // 1.0) else random.randint(w_min // 1.5, w_min // 1.0)
	r_y = (h_min // 1) if (h_min // 1.5) <= (h_min // 1.0) else random.randint(h_min // 1.5, h_min // 1.0)
	h, w, _ = img.shape
	try:
		rows = h // (y + r_y) + 1
		cols = w // (x + r_x) + 1
	except:
		print(results)
		print("y:{} r_y:{}, x:{}, r_x:{}".format(y, r_y, x, r_x))
		raise "error"
	for i in range(rows):
		y_start = i * (y + r_y)
		y_end = y_start + y
		for j in range(cols):
			x_start = j * (x + r_x)
			x_end = x_start + x
			if i == rows - 1 and j == cols - 1:
				h - y_start
				img[y_start:, x_start:, :] = np.random.randint(0, 255, (h-y_start, w-x_start, 3))
			elif i == rows - 1:
				img[y_start:, x_start:x_end, :] = np.random.randint(0, 255, (h-y_start, x_end-x_start, 3))
			elif j == cols - 1:
				img[y_start:y_end, x_start:, :] = np.random.randint(0, 255, (y_end-y_start, w-x_start, 3))
			else:
				img[y_start:y_end, x_start:x_end, :] = np.random.randint(0, 255, (y_end-y_start, x_end-x_start, 3))

	return img
