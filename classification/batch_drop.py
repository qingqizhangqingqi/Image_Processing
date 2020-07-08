import numpy.random as random
import torch.nn as nn

class BatchDrop(nn.Module):
	def __init__(self, h_ratio=0.4, w_ratio=0.4):
		super(BatchDrop, self).__init__()
		self.h_ratio = h_ratio
		self.w_ratio = w_ratio

	def forward(self, x):
		if self.training and random.random() > 0.5:
			h, w = x.size()[-2:]
			rh = int(self.h_ratio * h)
			rw = int(self.w_ratio * w)
			sx = random.randint(0, h-rh)
			sy = random.randint(0, w-rw)
			mask = x.new_ones(x.size())
			mask[:, :, sx:sx+rh, sy:sy+rw] = 0
			x = x * mask
		return x
