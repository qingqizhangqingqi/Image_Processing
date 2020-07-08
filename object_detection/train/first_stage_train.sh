#@/bin/bash

cd src/get_feature/
python setup.py develop
python tools/train.py configs/guided_anchoring/ga_htc.py --gpus=2 --gpu=0,1
 
