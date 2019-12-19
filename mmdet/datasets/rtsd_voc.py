"""
https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md
"""

#from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os.path as osp
import xml.etree.ElementTree as ET

#from rtsd_9_labels import ValidLabels
import numpy as np

from mmdet.datasets import CustomDataset, XMLDataset
from .coco import CocoDataset
from .registry import DATASETS

ValidLabels = [
    "danger",
    "mandatory",
    "prohibitory",
    "blue_border",
    "blue_rect",
    "main_road",
    "give_way",
    "white_rect",
    "end_prohibitory"
]


@DATASETS.register_module
class RTSD(XMLDataset):
    def __init__(self, ann_prefix: str, only_one_label=False, **kwargs):
        # Block bellow should lay before superconstructor
        self._ann_prefix = ann_prefix
        if only_one_label:
            self.CLASSES = ['object']

            # it's necessary hack for windows cause it was falling when serialize lambdas
            self._category_converter = self.convert_to_1_class

        else:
            self.CLASSES = ValidLabels

            # it's necessary hack for windows cause it was falling when serialize lambdas
            self._category_converter = self.leave_as_is

        super(RTSD, self).__init__(**kwargs)

        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}

    def convert_to_1_class(self, x):
        return 'object'

    def leave_as_is(self, x):
        return x

    def load_annotations(self, ann_file):
        img_infos = []
        img_names = mmcv.list_from_file(ann_file)
        fails = 0
        for filename in img_names:
            img_id = osp.splitext(filename)[0]
            xml_path = osp.join(self._ann_prefix, '{}.xml'.format(img_id))
            try:
                tree = ET.parse(xml_path)
            except FileNotFoundError:
                #print('no annotation ', xml_path)
                fails += 1
                continue
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        print('{} images with annotations, and {} without'.format(len(img_infos), fails))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self._ann_prefix, '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text

            name = self._category_converter(name)

            label = self.cat2label[name]

            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann