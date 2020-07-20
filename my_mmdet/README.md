## mmdetection的数据增强

`transforms.py` 和 `mixup_aug.py` 放在mmde/datasets/piepelines 并在`__init__`文件中导包。

使用时，在`voc0712.py`中的 train_pipeline 中添加对应的字典。

