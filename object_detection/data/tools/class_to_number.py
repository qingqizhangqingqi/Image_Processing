import os
import json
CLASSES = {'沾污':'1', '错花':'2', '水印':'3', '花毛':'4', '缝头':'5', '缝头印':'6', '虫粘':'7', '破洞':'8', '褶子':'9',
        '织疵':'10', '漏印':'11', '蜡斑':'12', '色差':'13', '网折':'14', '其他':'15'}


def change(file_path, save_path):
	file = open(file_path, "rb")
	file_json = json.load(file)
	for i, item in enumerate(file_json):
		name = item['name']
		defect_name = item['defect_name']
		item['defect_name'] = CLASSES[defect_name]

	with open(save_path, 'w') as fp:
		json.dump(file_json, fp, indent=4, separators=(',', ': '))	


if __name__ == "__main__":
	anno_files = os.listdir('json_anno')
	save_dir = "number"
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	for anno_file in anno_files:
		path = os.path.join('json_anno', anno_file)
		save_path = os.path.join(save_dir, anno_file)
		change(path, save_path)

