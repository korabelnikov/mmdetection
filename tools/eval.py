"""
In contrast to VOC eval this script produce Precision/Recall Curve by vary
detection threshold. Averaging applied by images.

"""
from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core.evaluation.mean_ap import get_cls_results, tpfp_default
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models.detectors.base import imshow_det_bboxes_with_gt
from tools.voc_eval import remove_images_from_pipeline


def eval(result_file, dataset, iou_thr=0.5, show=False):
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

        ann = dataset.get_ann_info(i)
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
            det_labels = np.zeros((len(det_results[i][0], )))
            imshow_det_bboxes_with_gt(img, det_bboxes, det_labels,
                                      gt_bboxes=bboxes,
                                      gt_labels=labels)
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = None
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    return eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=[(0, 16), (16, 24), (24, 32), (32, 64), (64, 256)],
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)



def eval_map(det_results,
             gt_bboxes,
             gt_labels,
             gt_ignore=None,
             scale_ranges=None,
             iou_thr=0.5,
             dataset=None,
             print_summary=True,
             thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                         0.6, 0.7, 0.8, 0.9]):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list): a list of list, [[cls1_det, cls2_det, ...], ...]
        gt_bboxes (list): ground truth bboxes of each image, a list of K*4
            array.
        gt_labels (list): ground truth labels of each image, a list of K array
        gt_ignore (list): gt ignore indicators of each image, a list of K array
        scale_ranges (list, optional): [(min1, max1), (min2, max2), ...]
        iou_thr (float): IoU threshold
        dataset (None or str or list): dataset name or dataset classes, there
            are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc.
        print_summary (bool): whether to print the mAP summary

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(gt_bboxes) == len(gt_labels)
    if gt_ignore is not None:
        assert len(gt_ignore) == len(gt_labels)
        for i in range(len(gt_ignore)):
            assert len(gt_labels[i]) == len(gt_ignore[i])
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1

    num_classes = len(det_results[0])  # positive class num
    gt_labels = [
        label if label.ndim == 1 else label[:, 0] for label in gt_labels
    ]

    metrics_method = precision_recall_over_detections
    #metrics_method = precision_recall_over_samples

    precisions, recalls = metrics_method(
        area_ranges, det_results, gt_bboxes, gt_ignore, gt_labels,
        iou_thr, num_classes, num_scales, thresholds)

    # todo this is only for sole class
    precisions = precisions[0]
    recalls = recalls[0]
    ##

    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt

    mAP = []
    precision_thr = [0.3, 0.5]
    recall_at = np.zeros((num_scales, len(precision_thr)))  # recalls
    # for each size group plot PR curve
    #plt.title("PR curves w.r.t size groups")
    plt.xlabel('precision')
    plt.ylabel('recall')
    for i, (recalls_i, precisions_i) in enumerate(zip(recalls, precisions)):
        plt.plot(precisions_i, recalls_i,
                 label="Size group {}".format(scale_ranges[i]))

        f = interp1d(precisions_i, recalls_i,
                     fill_value=(recalls_i[0], 0), bounds_error=False)
        # recall@prec
        for j, thr in enumerate(precision_thr):
            recall_at[i, j] = f(thr)
        # mAP
        mAP.append(np.mean([f(x) for x in np.linspace(0, 1, 100)]))
        #plt.plot(np.linspace(0, 1, 100), [f(x) for x in np.linspace(0, 1, 100)],
        #         label="int Size group {}".format(scale_ranges[i]))

    plt.legend()
    plt.show()

    # print_map_summary(mean_ap, eval_results, dataset, area_ranges)
    for i in range(len(area_ranges)):
        print('='*80)
        print('size group: \t\t', scale_ranges[i])
        print('mAP(100 p): {:.2f} \t\t'.format(mAP[i]*100))
        for r, p in zip(recall_at[i], precision_thr):
            print('Recall@{}: \t{:.2f} '.format(p, r*100), end='')
        print('')

    return mAP, recall_at


def precision_recall_over_samples(area_ranges, det_results, gt_bboxes,
                                  gt_ignore, gt_labels, iou_thr, num_classes,
                                  num_scales, thresholds):
    # calculate tp and fp for each image
    # 0 dim is class index
    # 1 dim size group index
    # 2 dim is img's index
    # 3 dim is thr's index
    # Оценка полученная таким способом занижена. It doesn
    # срабатывания в тех группах размеров где нет gt
    K = len(gt_labels)
    all_precisions = np.zeros((num_classes, num_scales, len(thresholds)))
    all_recalls = np.zeros((num_classes, num_scales, len(thresholds)))
    def area(a):
        return (a[2]-a[0])*(a[3]-a[1])

    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gt_ignore = get_cls_results(
            det_results, gt_bboxes, gt_labels, gt_ignore, i)
        for l, thr in enumerate(thresholds):
            precisions = [[] for _ in range(num_scales)]
            recalls = [[] for _ in range(num_scales)]
            for j in range(len(cls_dets)):  # by images
                cls_gt = cls_gts[j]
                cls_det = cls_dets[j]
                cls_det_thr = np.array([det for det in cls_det if det[4] > thr])
                if len(cls_gt) == 0:
                    pass
                elif len(cls_det_thr) == 0:
                    #continue
                    for s, sz in enumerate(area_ranges):
                        p = sum(1 for cls_gt_i in cls_gt if sz[0] < area(cls_gt_i) < sz[1])
                        if p == 0:
                            TREAT_0DIV0_AS_1 = False
                            if TREAT_0DIV0_AS_1:
                                precisions[s].append(1)
                                recalls[s].append(1)
                        else:
                            precisions[s].append(1)
                            recalls[s].append(0)
                else:
                    tp, fp, all_p = tpfp_default(cls_det_thr, cls_gt, cls_gt_ignore[j],
                                                 iou_thr, area_ranges)
                    # aggregate over detection inside the image
                    tps = np.sum(tp, axis=-1)
                    fps = np.sum(fp, axis=-1)

                    precision = tps / (tps + fps)
                    recall = tps / all_p

                    for s, p,r in zip(range(num_scales), precision, recall):
                        if ~np.isnan(p*r):
                            precisions[s].append(p)
                            recalls[s].append(r)
                        
            precisions = [np.mean(p) for p in precisions]
            recalls = [np.mean(r) for r in recalls]

            all_precisions[i, :, l] = precisions[:]
            all_recalls[i, :, l] = recalls[:]

    return all_precisions, all_recalls


def precision_recall_over_detections(area_ranges, det_results, gt_bboxes, gt_ignore, gt_labels, iou_thr, num_classes,
                                     num_scales, thresholds):
    # calculate tp and fp for each image
    # 0 dim is class index
    # 1 dim size group index
    # 2 dim is img's index
    # 3 dim is thr's index
    # Оценка полученная таким способом занижена. It doesn
    # срабатывания в тех группах размеров где нет gt
    K = len(gt_labels)
    precisions = np.zeros((num_classes, num_scales, len(thresholds)))
    recalls = np.zeros((num_classes, num_scales, len(thresholds)))
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gt_ignore = get_cls_results(
            det_results, gt_bboxes, gt_labels, gt_ignore, i)
        tps = np.zeros(num_scales)
        fps = np.zeros(num_scales)
        all_ps = np.zeros(num_scales)
        for l, thr in enumerate(thresholds):
            for j in range(len(cls_dets)):  # by images
                cls_gt = cls_gts[j]
                cls_det = cls_dets[j]
                cls_det_thr = np.array([det for det in cls_det if det[4] > thr])
                if len(cls_det_thr) == 0:

                    # TODO take into account images without gt
                    # is there a way to do it?
                    pass
                else:
                    tp, fp, all_p = tpfp_default(cls_det_thr, cls_gt, cls_gt_ignore[j],
                                                 iou_thr, area_ranges)
                    # aggregate over detection inside the image
                    tp = np.sum(tp, axis=-1)
                    fp = np.sum(fp, axis=-1)

                    tps += tp
                    fps += fp
                    all_ps += all_p

            precision = tps / (tps + fps)
            recall = tps / all_ps

            precisions[i, :, l] = precision[:]
            recalls[i, :, l] = recall[:]
    return precisions, recalls


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
    map, recalls = eval(args.result, test_dataset, args.iou_thr, args.show)


if __name__ == '__main__':
    main()
