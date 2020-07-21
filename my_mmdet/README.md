## mmdetection的数据增强

`transforms.py` 和 `mixup_aug.py` 放在mmde/datasets/piepelines 并在`__init__`文件中导包。

使用时，在`voc0712.py`中的 train_pipeline 中添加对应的字典。

RandomFlip 随机水平对称

RandomVFlip 随机垂直对称

RandomTranslate 随机平移

ContrastAndBrightness 对比度亮度变暗

AffineTransformation 逆时针选装90°

MotionBlur 运动模糊

GridMask 遮挡

Mixup 图片混合