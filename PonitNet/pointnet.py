# Copyright (c) Mike.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TNet(nn.Layer):
    def __init__(self, k=64, channels=3):
        super().__init__()
        self.conv1 = nn.Conv1D(channels, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.k = k

    def forward(self, x):

        B = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = paddle.eye(self.k).reshape([1, self.k * self.k]).tile([B, 1])

        x = x + iden
        x = x.reshape([-1, self.k, self.k])
        return x


class PointNetEncoder(nn.Layer):
    def __init__(
        self, global_feat=True, input_transform=True, feature_transform=False, channel=3
    ):
        super().__init__()

        self.global_feat = global_feat
        if input_transform:
            self.input_transfrom = TNet(k=channel)
        else:
            self.input_transfrom = lambda x: paddle.eye(
                channel, channel, dtype=paddle.float32
            )

        self.conv1 = nn.Conv1D(channel, 64, 1)
        self.conv2 = nn.Conv1D(64, 64, 1)
        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(64)

        if feature_transform:
            self.feature_transform = TNet(k=64)
        else:
            self.feature_transform = lambda x: paddle.eye(64, 64, dtype=paddle.float32)

        self.conv3 = nn.Conv1D(64, 64, 1)
        self.conv4 = nn.Conv1D(64, 128, 1)
        self.conv5 = nn.Conv1D(128, 1024, 1)
        self.bn3 = nn.BatchNorm1D(64)
        self.bn4 = nn.BatchNorm1D(128)
        self.bn5 = nn.BatchNorm1D(1024)

    def forward(self, x):
        x = x.transpose([0, 2, 1])
        B, D, N = x.shape
        trans_input = self.input_transfrom(x)
        x = paddle.transpose(x, (0, 2, 1))
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = paddle.bmm(x, trans_input)
        if D > 3:
            x = paddle.cat([x, feature], dim=2)
        x = paddle.transpose(x, (0, 2, 1))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        trans_feat = self.feature_transform(x)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.bmm(x, trans_feat)
        x = paddle.transpose(x, (0, 2, 1))

        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        if self.global_feat:
            return x, trans_input, trans_feat
        else:
            x = x.reshape((-1, 1024, 1)).repeat(1, 1, N)
            return paddle.cat([x, pointfeat], 1), trans_input, trans_feat


class PointNetCls(nn.Layer):
    def __init__(self, k=40, use_normals=True):
        super().__init__()
        if use_normals:
            channels = 6
        else:
            channels = 3
        self.feat = PointNetEncoder(
            global_feat=True,
            input_transform=True,
            feature_transform=True,
            channel=channel,
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1D(512)
        self.bn2 = nn.BatchNorm1D(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans_input, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_input, trans_feat


class PointNetPartSeg(nn.Layer):

    def __init__(self, part_num=50, use_normals=True):
        super().__init__()
        if use_normals:
            channels = 6
        else:
            channels = 3
        self.part_num = part_num
        self.stn = TNet(k=3, channels=channels)
        self.conv1 = nn.Conv1D(channels, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 128, 1)
        self.conv4 = nn.Conv1D(128, 512, 1)
        self.conv5 = nn.Conv1D(512, 2048, 1)
        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(128)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(2048)
        self.fstn = TNet(k=128, channels=128)
        self.convs1 = nn.Conv1D(4944, 256, 1)
        self.convs2 = nn.Conv1D(256, 256, 1)
        self.convs3 = nn.Conv1D(256, 128, 1)
        self.convs4 = nn.Conv1D(128, part_num, 1)
        self.bns1 = nn.BatchNorm1D(256)
        self.bns2 = nn.BatchNorm1D(256)
        self.bns3 = nn.BatchNorm1D(128)

    def forward(self, point_set, cls):
        point_set = point_set.transpose([0, 2, 1])
        B, D, N = point_set.shape
        trans = self.stn(point_set)  # B,3,3
        point_set = point_set.transpose([0, 2, 1])

        if D > 3:
            point_set, feature = point_set[:,:,:3], point_set[:,:,3:]
            
        point_set = paddle.bmm(point_set, trans)  # B,N,3 @ B,3,3
        if D > 3:
            point_set = paddle.concat([point_set, feature], axis=2)

        point_cloud = point_set.transpose([0, 2, 1])

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))  # B,64,N
        out2 = F.relu(self.bn2(self.conv2(out1)))  # B,128,N
        out3 = F.relu(self.bn3(self.conv3(out2)))  # B,128,N

        trans_feat = self.fstn(out3)  # B,128,128
        x = out3.transpose([0, 2, 1])
        net_transformed = paddle.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose([0, 2, 1])  # B,128,N

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))  # B,512,N
        out5 = self.bn5(self.conv5(out4))  # B,2048,N
        out_max = paddle.max(out5, 2, keepdim=True)  # B,2048,1
        out_max = out_max.reshape([-1, 2048])

        out_max = paddle.concat([out_max, cls], axis=1)  # B,2048 + B,16
        
        expand = out_max.reshape([-1, 2048+16, 1]).tile([1, 1, N])  # B 2064 N
        concat = paddle.concat([expand, out1, out2, out3, out4, out5], axis=1)  # B,4944,N
        
        seg_pred = F.relu(self.bns1(self.convs1(concat)))  
        seg_pred = F.relu(self.bns2(self.convs2(seg_pred)))  
        seg_pred = F.relu(self.bns3(self.convs3(seg_pred)))
        seg_pred = self.convs4(seg_pred)   # B 50 N
        seg_pred = seg_pred.transpose([0, 2, 1])  
        seg_pred = F.log_softmax(seg_pred.reshape([-1, self.part_num]), axis=-1)
        seg_pred = seg_pred.reshape([B, N, self.part_num]) # [B, N, 50]

        return seg_pred, trans_feat

if __name__ == "__main__":
    
    m = PointNetPartSeg()
    x = paddle.randn([4,2500,3])
    label = paddle.randn([4,1,16])
    m(x,label)

