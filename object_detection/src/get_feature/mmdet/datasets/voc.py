from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class VOCDataset(XMLDataset):
    CLASSES = ('沾污', '错花', '水印', '花毛', '缝头', '缝头印', '虫粘', '破洞', '褶子',
        '织疵', '漏印', '蜡斑', '色差', '网折', '其他', '正常')
    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            self.year = 2012
