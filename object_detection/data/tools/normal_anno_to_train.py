import os
import random

new_lines = []
normal_annos = os.listdir('normal_anno')
for normal_anno in normal_annos:
	base = normal_anno.split('.')[0] + '\n'
	new_lines.append(base)

	
f = open('ImageSets/Main/train.txt', 'r')
lines = f.read().splitlines()
for line in lines:
	new_lines.append(line + '\n')

f.close()
new_f = open('ImageSets/Main/train_with_normal.txt', 'w')
random.shuffle(new_lines)
for line in new_lines:
	new_f.write(line)

new_f.close()
