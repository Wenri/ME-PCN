from __future__ import print_function

import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

from raypoint import sample_rays, RayClsSample, WeightNetHidden, sample_from_grouped_rays, sample_rays_vec
from utils import import_file

file_path = os.path.dirname(os.path.realpath(__file__))
expansion = import_file('expansion_penalty_module',
                        os.path.join(file_path, 'expansion_penalty', 'expansion_penalty_module.py'))


class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))
        ).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=8192, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class PointNetfeatVarlen(nn.Module):
    def __init__(self, num_points=8192, global_feat=True):
        super(PointNetfeatVarlen, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(9, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x, cnt):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.split(x, cnt.cpu().numpy().tolist(), dim=2)
        x = torch.cat([torch.max(a, 2)[0] for a in x])
        return x


class PointNetfeatRayconv(nn.Module):
    def __init__(self, num_points=8192, input_dim=10, global_feat=True):
        super(PointNetfeatRayconv, self).__init__()
        self.input_dim = input_dim
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(input_dim, 16, 1)
        self.conv2 = torch.nn.Conv1d(16, 32, 1)
        self.conv3 = torch.nn.Conv1d(32, 32, 1)

        self.bn1 = torch.nn.BatchNorm1d(16)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.weight_hidden = WeightNetHidden(input_dim + 1, [16, 32, 32])
        self.fc1 = torch.nn.Linear(1024, 32)
        self.bn4 = torch.nn.BatchNorm1d(32)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x, mask):
        batchsize = mask.size()[0]

        x = x.view(batchsize, -1, self.input_dim).transpose(1, 2)
        infeat = x.view(batchsize, -1, mask.shape[1], mask.shape[2])
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.view(batchsize, -1, mask.shape[1], mask.shape[2]).transpose(1, 2)
        infeat = torch.cat((infeat, mask.unsqueeze(1).float()), dim=1)
        weight = self.weight_hidden(infeat).view(batchsize, 32, -1)
        weight = weight.transpose(1, 2).view(batchsize, mask.shape[1], mask.shape[2], -1)
        x = torch.matmul(x, weight).view(batchsize, mask.shape[1], -1)
        x = self.fc1(x).transpose(1, 2)
        return self.bn4(x)


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class PointNetRes(nn.Module):
    def __init__(self, input_dim=68):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1152, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x


class MSN(nn.Module):
    OUTPUT_TUPLE = namedtuple('MSNOut', 'output1 output2 ray, mask, ray_s1, mask_s1, ray_s2, mask_s2 expansion_penalty')

    def __init__(self, num_points=8192, bottleneck_size=1024, n_primitives=16):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.encoder = PointNetfeat(num_points, global_feat=True)
        self.raycoderv2 = PointNetfeatVarlen(num_points, global_feat=True)
        self.rayseq = nn.Sequential(
            nn.Linear(2048, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()
        self.rayfeat1 = PointNetfeatRayconv(num_points, 9, global_feat=True)
        self.rayfeat2 = PointNetfeatRayconv(num_points, 15, global_feat=True)
        self.rayclssample = RayClsSample()

        plus_primitives = n_primitives + 1
        points_by_primitives = num_points // n_primitives
        labels_generated_points = torch.arange(1, plus_primitives * points_by_primitives + 1).view(
            points_by_primitives, plus_primitives).transpose(0, 1)
        labels_generated_points = labels_generated_points % plus_primitives
        self.labels_generated_points = labels_generated_points.contiguous().view(-1)

    def forward(self, x, pc_end, ray_start, ray_end, m_trans, n_nrays=8, dist_gap=0.05):
        batchsize = x.size()[0]
        partial = x
        ray1, mask1 = sample_rays(dist_gap, n_nrays, x.transpose(1, 2), ray_start, ray_end, m_trans)
        x = torch.cat((x, torch.count_nonzero(mask1, dim=-1).unsqueeze(1) / n_nrays), dim=1)
        x = self.encoder(x)

        ray1m = ray1[mask1]
        ray1_cnt = torch.count_nonzero(mask1.view(batchsize, -1), dim=-1)
        ray1_vec = sample_from_grouped_rays(ray1m, ray1_cnt)
        ray = self.raycoderv2(ray1m.T.unsqueeze(0), ray1_cnt)
        x = self.rayseq(torch.cat([x, ray], dim=-1))
        outs = []
        for i in range(0, self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.n_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))

        outs = torch.cat(outs, 2).contiguous()
        out1 = outs.transpose(1, 2).contiguous()

        dist, _, mean_mst_dis = self.expansion(out1, self.num_points // self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)

        out1_ray, out1_mask = sample_rays(dist_gap, n_nrays, out1, ray_start, ray_end, m_trans)
        out2_ray, out2_mask = sample_rays_vec(dist_gap, n_nrays, out1, partial, pc_end, ray1_vec, m_trans)
        out1_feat = self.rayfeat1(out1_ray, out1_mask)
        out2_feat = self.rayfeat2(out2_ray, out2_mask)
        out1_feat = torch.cat((out1_feat, out2_feat), dim=1)
        xx = self.rayclssample(partial, outs, out1_feat, mean_mst_dis)

        delta = self.res(xx)
        xx = xx[:, 0:3, :]
        out2 = (xx + delta).transpose(2, 1).contiguous()

        return self.OUTPUT_TUPLE(out1, out2, ray1, mask1, out1_ray, out1_mask, out2_ray, out2_mask, loss_mst)
