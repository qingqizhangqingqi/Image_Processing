import xml.etree.ElementTree as ET
import os
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--image_dir', type=str, default='defect', help='image dir')
    parser.add_argument('--anno_dir', type=str, default='Annotations', help='annotation dir')
    args = parser.parse_args()
    return args

def indent(elem, level=0):
        i = "\n" + level*"\t"
        if len(elem):
                if not elem.text or not elem.text.strip():
                        elem.text = i + "\t"
                if not elem.tail or not elem.tail.strip():
                        elem.tail = i
                for elem in elem:
                        indent(elem, level+1)
                if not elem.tail or not elem.tail.strip():
                        elem.tail = i
        else:   
                if level and (not elem.tail or not elem.tail.strip()):
                        elem.tail = i


def create_xml(anno_dir, image_dir, name, defect_names, bboxes, width, height): 
        name_prefix = name.split('.')[0] 
        xml_name = name_prefix + ".xml" 
        xml_path = os.path.join(anno_dir, xml_name) 
 
        root = ET.Element('annotation') 
        xml_jpg_name = ET.SubElement(root, 'filename') 
        xml_jpg_name.text = name 
        xml_image_path = ET.SubElement(root, 'path') 
        xml_image_path.text = os.path.join(image_dir, name_prefix, name) 
        xml_size = ET.SubElement(root, 'size') 
        xml_width = ET.SubElement(xml_size, 'width') 
        xml_width.text = str(width) 
        xml_height = ET.SubElement(xml_size, 'height') 
        xml_height.text = str(height) 
        xml_depth = ET.SubElement(xml_size, 'depth') 
        xml_depth.text = '3' 
        for defect_name, bbox in zip(defect_names, bboxes): 
                xmin = bbox[0] 
                ymin = bbox[1] 
                xmax = bbox[2] 
                ymax = bbox[3] 
 
                xml_object = ET.SubElement(root, 'object') 
                xml_object_diff = ET.SubElement(xml_object, 'difficult') 
                xml_object_diff.text = '0' 
                xml_object_name = ET.SubElement(xml_object, 'name') 
                xml_object_name.text = defect_name 
                xml_object_bndbox = ET.SubElement(xml_object, 'bndbox') 
                xml_bndbox_xmin = ET.SubElement(xml_object_bndbox, 'xmin') 
                xml_bndbox_xmin.text = str(xmin) 
                xml_bndbox_ymin = ET.SubElement(xml_object_bndbox, 'ymin') 
                xml_bndbox_ymin.text = str(ymin) 
                xml_bndbox_xmax = ET.SubElement(xml_object_bndbox, 'xmax') 
                xml_bndbox_xmax.text = str(xmax) 
                xml_bndbox_ymax = ET.SubElement(xml_object_bndbox, 'ymax') 
                xml_bndbox_ymax.text = str(ymax) 
 
 
        indent(root) 
        tree = ET.ElementTree(root) 
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def compute_iou(rec1, rec2):
	# computing area of each rectangles
	S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
	S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
	# computing the sum_area
	sum_area = S_rec1 + S_rec2
 
	# find the each edge of intersect rectangle
	left_line = max(rec1[1], rec2[1])
	right_line = min(rec1[3], rec2[3])
	top_line = max(rec1[0], rec2[0])
	bottom_line = min(rec1[2], rec2[2])
 
	# judge if there is an intersect
	if left_line >= right_line or top_line >= bottom_line:
		return 0
	else:
		intersect = (right_line - left_line) * (bottom_line - top_line)
		return (intersect / (sum_area - intersect))*1.0

def slip_window(anno_dir, image_dir, save_image_dir, save_anno_dir, min_size=4):
	rows = 2
	cols = 2
	anno_files = os.listdir(anno_dir)
	count = len(anno_files)
	for index, anno_file in enumerate(anno_files):
		image_base = anno_file.split('.')[0]
		style = image_base.split('_')[0]
		image_path = os.path.join(image_dir, image_base, image_base + '.jpg')
		template_path = os.path.join(image_dir, image_base, "template_{}.jpg".format(style))
		im = cv2.imread(image_path)
		template = cv2.imread(template_path)		
		height, width, _ = im.shape
		sub_height, sub_width = int(height/rows), int(width/cols)

		anno_path = os.path.join(anno_dir, anno_file)
		root = ET.parse(anno_path)
		objs = root.findall('object')

		for row in range(rows):
			for col in range(cols):
				width_start = col * sub_width
				width_end = (col+1) * sub_width
				height_start = row * sub_height
				height_end = (row+1) * sub_height

				sub_objs = []
				sub_obj_names = []
				new_im = im[height_start:height_end, width_start:width_end, :]
				new_template = template[height_start:height_end, width_start:width_end, :]
				for obj in objs:
					defect_name = obj.find('name').text
					bndbox = obj.find('bndbox')
					xmin = float(bndbox.find('xmin').text)
					ymin = float(bndbox.find('ymin').text)
					xmax = float(bndbox.find('xmax').text)
					ymax = float(bndbox.find('ymax').text)
					if xmin > width_end or ymin > height_end or xmax < width_start or ymax < height_start:
						continue

					if xmin < width_start:
						xmin = 0
					else:
						xmin -= width_start

					if ymin < height_start:
						ymin = 0
					else:
						ymin -= height_start

					if xmax > width_end:
						xmax = sub_width
					else:
						xmax -= width_start

					if ymax > height_end:
						ymax = sub_height
					else:
						ymax -= height_start

					obj_w = xmax - xmin
					obj_h = ymax - ymin
					if obj_w < min_size or obj_h < min_size:
						print("obj_w = {}, obj_h = {}".format(obj_w, obj_h))
						continue

					bbox = [xmin, ymin, xmax, ymax]
					sub_objs.append(bbox)
					sub_obj_names.append(defect_name)

				if len(sub_objs) <= 0:
					continue

				new_image_name = "{}{}{}.jpg".format(image_base, row, col)
				create_xml(save_anno_dir, save_image_dir, new_image_name, sub_obj_names, sub_objs, width_end-width_start, height_end-height_start)				
				image_save_path = os.path.join(save_image_dir, "{}{}{}".format(image_base, row, col))
				if not os.path.exists(image_save_path):
					os.mkdir(image_save_path)

				image_save_sub_path = os.path.join(image_save_path, new_image_name)
				template_save_path = os.path.join(image_save_path, "template_{}.jpg".format(style))
				cv2.imwrite(image_save_sub_path, new_im)
				cv2.imwrite(template_save_path, new_template)
		print("{}/{}".format(index, count))
	print('finished')

if __name__ == "__main__":
	args = parse_args()
	slip_window(args.anno_dir, args.image_dir, args.image_dir, args.anno_dir)
	#slip_window('normal_anno', 'normal', save_image_dir, save_xml_dir)


