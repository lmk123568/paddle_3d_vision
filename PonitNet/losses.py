# Copyright (c) Mike.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ClsLoss(nn.Layer):
    def __init__(self, mat_diff_loss_scale=1e-3):
        super().__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat=None):
        loss = F.cross_entropy(pred, target)

        if trans_feat is None:
            mat_diff_loss = 0
        else:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

class PartSegLoss(nn.Layer):
    def __init__(self, mat_diff_loss_scale=0.001):
        super().__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.cross_entropy(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


def feature_transform_reguliarzer(trans):
    d = trans.shape[1]
    I = paddle.eye(d)
    loss = paddle.mean(
        paddle.norm(
            paddle.bmm(trans, paddle.transpose(trans, (0, 2, 1))) - I, axis=(1, 2)
        )
    )
    return loss
