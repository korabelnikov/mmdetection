import logging
import numpy as np
from collections import OrderedDict

from mmcv.runner import load_state_dict

from ..registry import BACKBONES
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.squeezenet import Fire, init
from ..registry import HEADS

EPSILON = 1e-7


@HEADS.register_module
class SqueezeDet(nn.Module):
    def __init__(self, in_channels=512, anchor_per_grid=9, num_classes=80 + 1,
                 width=1248, height=384):
        super(SqueezeDet, self).__init__()
        self.anchors_per_grid = anchor_per_grid
        self.num_classes = num_classes - 1  # it's important

        self.width = width
        self.height = height

        num_output = anchor_per_grid * (self.num_classes + 1 + 4)
        self.features = nn.Sequential(
            Fire(in_channels, 16, 384, 384),
            Fire(384 + 384, 16, 384, 384),
            # dropout11 = torch.nn.dropout(fire11, self.keep_prob, name='drop11')
            nn.Conv2d(384 + 384, num_output, kernel_size=3, stride=1,
                      padding_mode='same', padding=1)
        )
        self.anchor_box = anchors(width, height)
        self.nanchors = len(self.anchor_box)
        self.num_objects = None

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.features[-1]:
                    init.normal_(m.weight, mean=0.0, std=0.0001)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, feats):
        preds = self.features(feats[-1])

        # probability
        num_class_probs = self.anchors_per_grid * self.num_classes
        pred_class_probs = torch.reshape(preds[:, :num_class_probs, :, :],
                                         [-1, self.num_classes])
        pred_class_probs = torch.softmax(pred_class_probs, dim=-1)
        pred_class_probs = torch.reshape(pred_class_probs,
                                         [-1, self.nanchors, self.num_classes])

        # confidence
        num_confidence_scores = self.anchors_per_grid + num_class_probs
        pred_conf = torch.reshape(preds[:,
                                  num_class_probs:num_confidence_scores, :, :],
                                  [-1, self.nanchors])
        pred_conf = torch.sigmoid(pred_conf)

        # bbox_delta
        pred_box_delta = torch.reshape(preds[:, num_confidence_scores:, :, :],
                                       [-1, self.nanchors, 4])

        # number of object. Used to normalize bbox and classification loss
        self.num_objects = torch.sum(self.input_mask)

        delta_x, delta_y, delta_w, delta_h = torch.unbind(pred_box_delta,
                                                          dim=2)

        anchor_x = self.anchor_box[:, 0]
        anchor_y = self.anchor_box[:, 1]
        anchor_w = self.anchor_box[:, 2]
        anchor_h = self.anchor_box[:, 3]

        box_center_x = anchor_x + delta_x * anchor_w
        box_center_y = anchor_y + delta_y * anchor_h
        EXP_THRESH = 1.0
        box_width = anchor_w * safe_exp(delta_w, EXP_THRESH)
        box_height = anchor_h * safe_exp(delta_h, EXP_THRESH)

        # trimming
        xmins, ymins, xmaxs, ymaxs = bbox_transform(
            [box_center_x, box_center_y, box_width, box_height])

        # The max x position is self.width - 1 since we use zero-based
        # pixels. Same for y.

        xmins = torch.min(torch.max(xmins, 0.0), self.width - 1)

        ymins = torch.min(torch.max(ymins, 0.0), self.height - 1.0)

        xmaxs = torch.max(torch.min(self.width - 1.0, xmaxs), 0.0)

        ymaxs = torch.max(torch.min(self.height - 1.0, ymaxs), 0.0)

        inv = bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])
        self.det_boxes = torch.stack(inv).permute((1, 2, 0))

        # tf.assign
        self.ious = _tensor_iou(
            bbox_transform(torch.unbind(self.det_boxes, dim=2)),
            bbox_transform(torch.unbind(self.box_input, dim=2)),
            self.input_mask, self.nanchors
        )
        # probs
        probs = torch.reshape(pred_conf, [-1, self.nanchors, 1])
        probs = pred_class_probs * probs

        self.det_probs = torch.max(probs, 2)
        self.det_class = torch.argmax(probs, 2)

    def loss(self):
        # loss coefficient for confidence regression
        LOSS_COEF_CONF = 1.0

        # loss coefficient for classification regression
        LOSS_COEF_CLASS = 1.0

        # loss coefficient for bounding box regression
        LOSS_COEF_BBOX = 10.0

        LOSS_COEF_BBOX = 5.0
        LOSS_COEF_CONF_POS = 75.0
        LOSS_COEF_CONF_NEG = 100.0
        LOSS_COEF_CLASS = 1.0

        # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
        # add a small value into log to prevent blowing up
        self.class_loss = torch.sum(
            (self.labels * (-torch.log(self.pred_class_probs + EPSILON))
             + (1 - self.labels) * (-torch.log(1 - self.pred_class_probs + EPSILON)))
            * self.input_mask * LOSS_COEF_CLASS) / self.num_objects

        input_mask = torch.reshape(self.input_mask, [-1, self.nanchors])
        self.conf_loss = torch.mean(
            torch.sum(
                (self.ious - self.pred_conf) ** 2
                * (input_mask * LOSS_COEF_CONF_POS / self.num_objects
                   + (1 - self.input_mask) * LOSS_COEF_CONF_NEG / (self.nanchors - self.num_objects)),
                dim=1
            ), )
        self.bbox_loss = torch.sum(
            LOSS_COEF_BBOX * (
                    self.input_mask * (self.pred_box_delta - self.box_delta_input)) ** 2
        ) / self.num_objects

        # add above losses as well as weight decay losses to form the total loss
        self.loss = self.class_loss + self.conf_loss + self.bbox_loss
        return self.loss


def anchors(IMAGE_WIDTH, IMAGE_HEIGHT):
    #H, W, B = 24, 78, 9
    H, W, B = 24, 44, 9
    anchor_shapes = np.reshape(
        [np.array(
            [[36., 37.], [366., 174.], [115., 59.],
             [162., 87.], [38., 90.], [258., 173.],
             [224., 108.], [78., 170.], [72., 43.]])] * H * W,
        (H, W, B, 2)
    )
    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W + 1) * float(IMAGE_WIDTH) / (W + 1)] * H * B),
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )
    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H + 1) * float(IMAGE_HEIGHT) / (H + 1)] * W * B),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )
    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )

    return anchors


def bbox_transform(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]] * 4
    out_box[0] = cx - w / 2
    out_box[1] = cy - h / 2
    out_box[2] = cx + w / 2
    out_box[3] = cy + h / 2

    return out_box


def bbox_transform_inv(bbox):
    """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
    for numpy array or list of tensors.
    """
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]] * 4

    width = xmax - xmin + 1.0
    height = ymax - ymin + 1.0
    out_box[0] = xmin + 0.5 * width
    out_box[1] = ymin + 0.5 * height
    out_box[2] = width
    out_box[3] = height

    return out_box


def safe_exp(w, thresh):
    """Safe exponential function for tensors."""

    slope = np.exp(thresh)
    lin_bool = w > thresh
    lin_region = lin_bool.float()

    lin_out = slope * (w - thresh + 1.)
    exp_out = torch.exp(torch.where(lin_bool, torch.zeros_like(w), w))

    out = lin_region * lin_out + (1. - lin_region) * exp_out
    return out


def _tensor_iou(box1, box2, input_mask, nanchors):
    # intersection
    xmin = torch.max(box1[0], box2[0])
    ymin = torch.max(box1[1], box2[1])
    xmax = torch.min(box1[2], box2[2])
    ymax = torch.min(box1[3], box2[3])

    w = torch.max(xmax - xmin, 0.0)
    h = torch.max(ymax - ymin, 0.0)
    intersection = w * h

    # union
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    union = w1 * h1 + w2 * h2 - intersection
    return intersection / (union + EPSILON) * \
           torch.reshape(input_mask, [-1, nanchors])
