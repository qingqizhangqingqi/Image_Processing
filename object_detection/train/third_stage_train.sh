#@/bin/bash

cd src/feature_substract_detection/
python tools/generate_normal_gt.py --config configs/guided_anchoring/ga_split_window_with_normal.py --model work_dirs/get_split_window_with_normal/epoch_17.pth

cd ../../data/
python tools/split_window.py --image_dir normal --anno_dir normal_anno/
cp -rf normal/* defect/
cp -rf normal_anno/* Annotations/
cd ../src/feature_substract_detection/
python tools/train.py configs/guided_anchoring/ga_with_normal.py --resume_from work_dirs/get_split_window_with_normal/epoch_17.pth --gpus=2 --gpu=0,1
mv work_dirs/get_split_window_with_normal/epoch_24.pth ../../weights/
