from free_gpu import get_free_gpu
import os
import time
import subprocess

if __name__ == "__main__":
	while True:
		get_free_gpu(14)
		epoch_files = os.listdir('work_dirs/ga_fp16_32')
		max_epoch = 0
		for epoch_file in epoch_files:
			if not'epoch_' in epoch_file:
				continue

			epoch = int(epoch_file.split('.')[0].split('_')[1])
			if epoch > max_epoch:
				max_epoch = epoch

		print("max epoch = {}".format(max_epoch))
		if max_epoch == 0:
			cmd = 'python tools/train.py configs/guided_anchoring/ga_fp16_32.py --gpu=1 --gpus=1'
		else:
			cmd = 'python tools/train.py configs/guided_anchoring/ga_fp16_32.py --resume_from=work_dirs/ga_fp16_32/epoch_{}.pth --gpus=1 --gpu=1'.format(max_epoch)
		p = subprocess.call(cmd, shell=True)
		#print(p.pid)
		
		#os.system(cmd)
		time.sleep(10)	

