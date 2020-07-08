#@/bin/bash

cd src/feature_substract_detection/
python tools/inference.py --config configs/guided_anchoring/ga_with_normal.py --model ../../weights/epoch_24.pth --gpu 0 --data ../../data/test_b/
