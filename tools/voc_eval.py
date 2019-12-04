from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.datasets.pipelines import LoadImageInfo
from mmdet.models.detectors.base import imshow_det_bboxes_with_gt


def voc_eval(result_file, dataset, iou_thr=0.5, show=False):
    det_results = mmcv.load(result_file)

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)

    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i, data in zip(range(len(dataset)), data_loader):
        bboxes = data['gt_bboxes'].data[0]
        labels = data['gt_labels'].data[0]

        bboxes = np.concatenate(bboxes)
        labels = np.concatenate(labels)

        ann = dataset.get_ann_info(i)  # TODO rem it
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])

        if show:
            img = data['img'].data[0]
            img = img.cpu().numpy()[0].transpose(1, 2, 0)
            img = np.ascontiguousarray(img)
            img -= img.min()
            img /= img.max()

            det_bboxes = np.vstack(det_results[i])
            det_labels = np.zeros((len(det_results[i][0],)))
            imshow_det_bboxes_with_gt(img, det_bboxes, det_labels, gt_bboxes=bboxes, gt_labels=labels)
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = None
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=[(0, 16), (16, 24), (24, 32), (32, 64), (64, 256)],
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)


def remove_images_from_pipeline(test_dataset):
    test_dataset.annotations_only_mode()
    # assert todo
    test_dataset.pipeline.transforms[0] = LoadImageInfo()
    # todo assert normolize
    del test_dataset.pipeline.transforms[4]
    collect = test_dataset.pipeline.transforms[-1]
    assert collect.keys[0] == 'img'
    del collect.keys[0]
    meta_keys = collect.meta_keys[:-1]
    collect.meta_keys = meta_keys


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument('--show',
                        dest='show',
                        action='store_true',
                        default=False,
                        help='Show all detections on images')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.val, datasets)
    if not args.show:
        remove_images_from_pipeline(test_dataset)
    voc_eval(args.result, test_dataset, args.iou_thr, args.show)


if __name__ == '__main__':
    main()
