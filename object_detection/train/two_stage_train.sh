#@/bin/bash

cd src/feature_substract_detection/
python setup.py develop
python tools/get_backbone_feature.py
python tools/train.py configs/guided_anchoring/ga_split_window_with_normal.py --gpus=2 --gpu=0,1
