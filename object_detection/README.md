##linux系统需支持中文编码##

使用的深度学习框架为 pytorch
pytorch 版本1.2.0
cuda版本10.1.168， cudnn版本7.6


1.框架依赖库安装
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install cython
cd src/mmcv
pip install -e .

cd ../get_feature
python setup.py develop

cd ../feature_substract_detection
python setup.py develop


1.解压数据&数据预处理
	cd ../../data
	mkdir defect
	mkdir normal
	mkdir json_anno
	mv guangdong1_round2_train*/defect/* defect
	mv guangdong1_round2_train*/normal/* normal/
	mv guangdong1_round2_train_part1_20190924/Annotations/anno_train.json json_anno/anno_train1.json
	mv guangdong1_round2_train2_20191004_Annotations/Annotations/anno_train.json json_anno/anno_train2.json
	rm -rf guangdong1_round2_train*
	# 数据格式转为VOC格式
	python tools/to_voc.py

	# 数据增强，将 GT 个数超过20个的图片，把 GT 抠出来20个为一组和 normal 或者 template 做 mixup，产生新的图片，并对目标较少的类别做数据增强，平衡类别数量（做法和多GT的相同）。
	python tools/aug_image.py

	# 切图，由于原始图片较大，且部分目标较小，在图片缩放之后目标太小了，所以做了切图操作。将原始图片切成2X2张图片，对于目标被切分的情况，如果被切的目标在切图中的长度或者宽度小于4，当做没有目标，反之，这个被切的目标也标注为目标。（切图和原始图都会送入网络训练，原始图将被缩放）
	python tools/split_window.py

	# 重新生成 trainval.txt 文件
	python tools/generate_train.py


预训练模型下载链接：
https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d-a5af3160.pth
mmdetection框架会自行下载此预训练模型，如果出现下载失败，请用其他方式下载预训练模型，并将下载的预训练模型拷贝到
~/.cache/torch/checkpoints/

3.训练
	# 一阶段训练，获得特征提取网络的backbone
	cd ../
	sh train/first_stage_train.sh

	# 二阶段训练，提取一阶段训练得到的backbone，对defect图片和template图片分别提取特征后，对特征相减，再执行目标检测
	sh train/two_stage_train.sh

	# 三阶段训练，使用二阶段训练得到的模型，对normal进行检测，将检测出的目标（即误检的目标）打上‘16’的伪标签后分割图片，将它们加入到训练集。提升学习率继续学习8个epoch
	sh train/third_stage_train.sh


4.预测
	# 通过切分图片进行预测，将图片切分为2*2张图片，进行预测，再将原图缩放到切割后的图片大小，进行检测，将检测结果bbox的宽高小于48的目标删除（小目标的检测主要在切图中预测，原图缩放太多，小目标检测不准）。切分的图片的检测结果相加，再和原图的检测结果做NMS，删除标签为色差，且宽高小于原图大小的0.9倍的目标（根据观察，色差都是整张图片的，色差面积过小极可能是在切分图片的误检，最高成绩上没有做这步操作，在验证集中map提升较多，最后一次提交模型提交错了。故看不出在测试集的效果）。
	sh inference/inference.sh

5.创新性
	1、改进目标检测网络，对defect图片和template图片进行特征相减之后再做目标检测，大大提高了map和acc、
	2、通过检测normal图片，对误检的目标做伪标签的方法，略微提高了map和acc
	3、通过使用切图和全图同时检测的方式，在切图上提高小目标的检测能力，在全图上检测大目标
	4、通过将GT特别多的图片，将GT抠出来mix到normal图片上，解决gt太多导致显存不足的问题。通过mix的方法，平衡类别数量不均衡的问题
