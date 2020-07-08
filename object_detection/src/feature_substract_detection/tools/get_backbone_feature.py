import torch

def print_pre_trained():
	name = '/home/lindeshou/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth'
	model = torch.load(name)
	for dicts in model:
		print(dicts)
	#dicts = model['state_dict']
	#for dict in dicts:
		#print(dict)
	#print(type(model))

def print_backbone():
	name = 'feature_get.pth'
	model = torch.load(name)
	#print(model)
	dicts = model["state_dict"]
	for dict in dicts:
		print(dict)


if __name__ == "__main__":
	#print_pre_trained()
	#print_backbone()
	model_name = '../get_feature/work_dirs/get_feature/epoch_17.pth'
	#model_name = 'work_dirs/best/epoch_20.pth'
	model = torch.load(model_name)

	backbone = dict()
	backbone['state_dict'] = dict()
	model_dicts = model['state_dict']
	for model_key in model_dicts:
		if not 'backbone' in model_key:
			continue

		key = model_key.split('backbone.')[1]
		#print("model_key = {}, key = {}".format(model_key, key))
		backbone['state_dict'][key] = model_dicts[model_key]

	torch.save(backbone, 'pre_trained/pre_trained_with_split_img.pth')
