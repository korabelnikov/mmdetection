import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import AnchorGenerator, anchor_target, multi_apply
from ..losses import smooth_l1_loss
from ..registry import HEADS
from .anchor_head import AnchorHead
import numpy as np


# TODO: add loss evaluator for SSD
@HEADS.register_module
class SSDHead2(AnchorHead):

    def __init__(self,
                 input_size=(1280, 800),
                 num_classes=81,
                 in_channels=(64, 128, 256, 512),
                 anchor_strides=(4, 8, 16, 32),  # should be orig_size/feat_size[i]
                 basesize_ratio_range=(0.1, 0.9),
                 #anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 anchor_ratios=(0.5, 1.0),#, 2.0),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 ):
        super(AnchorHead, self).__init__()
        # self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        num_anchors = len(anchor_ratios)
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    (num_anchors + 1) * 4, # TODO HACK!!!
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    (num_anchors + 1) * num_classes, # TODO HACK!!!
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)  # in percents
        max_ratio = int(max_ratio * 100)  # in percents
        #step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 1))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(np.array(input_size) * r // 100)
            max_sizes.append(np.array(input_size) * (r + step) // 100)

        # custom SSD 8 16 24 32 64 128№
        #print('custom SSD head used')
        #min_sizes = [8, 16, 24, 32, 64]
        #max_sizes = [16, 24, 32, 64, 128]

        min_sizes.insert(0, np.array(input_size) * 4 // 100)
        max_sizes.insert(0, np.array(input_size) * 10 // 100)

        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            stride = anchor_strides[k]
            ctr = ((stride - 1) / 2., (stride - 1) / 2.) #TODO WTF?

            # TODO NEED TO USE BOTH x and y scales and mb scale=(1, size_x/size_y)?
            scales_x = [1., np.sqrt(max_sizes[k] / min_sizes[k])[0]]

            anchor_generator = AnchorGenerator(
                base_size, scales_x, anchor_ratios, scale_major=False,
                ctr=ctr)
            indices = list(range(num_anchors))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(
                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.fp16_enabled = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        assert len(feats) == len(self.in_channels), 'incorrect number of features'
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            sampling=False,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
