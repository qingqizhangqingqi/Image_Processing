from free_gpu import get_free_gpu
import os
import time
import subprocess

if __name__ == "__main__":
	while True:
		get_free_gpu(14)
		epoch_files = os.listdir('work_dirs/htc')
		max_epoch = 0
		for epoch_file in epoch_files:
			if not'epoch_' in epoch_file:
				continue

			epoch = int(epoch_file.split('.')[0].split('_')[1])
			if epoch > max_epoch:
				max_epoch = epoch

		print("max epoch = {}".format(max_epoch))
		if max_epoch == 0:
			cmd = 'python tools/train.py configs/htc/htc.py --gpus=2'
		else:
			cmd = 'python tools/train.py configs/htc/htc.py --resume_from=work_dirs/htc/epoch_{}.pth --gpus=2'.format(max_epoch)
		p = subprocess.call(cmd, shell=True)
		#print(p.pid)
		
		#os.system(cmd)
		time.sleep(10)	

