import itertools
import os

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from torch import nn

from io_util.pc import resample_pcd
from utils import import_file

file_path = os.path.dirname(os.path.realpath(__file__))
MDS_module = import_file('MDS_module', os.path.join(file_path, 'MDS', 'MDS_module.py'))


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def sample_from_view(q_xyz, ray_start, m_trans, nsample, return_pts=False):
    batch_size = ray_start.shape[0]
    npoint = q_xyz.shape[1]

    m_proj, m_modeview = m_trans[:, 0], m_trans[:, 1]
    m_combine = torch.matmul(m_modeview, m_proj)
    q_xyzw = torch.cat((q_xyz, torch.ones(*q_xyz.shape[:2], device=q_xyz.device).unsqueeze(-1)), dim=-1)
    q_xyzw = torch.matmul(q_xyzw, m_combine)
    q_xyzw /= q_xyzw[..., 3].unsqueeze(-1)
    ray_pos = torch.matmul(ray_start, m_combine)
    ray_pos /= ray_pos[..., 3].unsqueeze(-1)

    point_indices = knn_points(q_xyzw[..., :2], ray_pos[..., :2], K=nsample).idx.reshape(-1)
    batch_indices = torch.repeat_interleave(torch.arange(batch_size), npoint * nsample)

    if not return_pts:
        return batch_indices, point_indices

    q_xyzw = q_xyzw[batch_indices, point_indices].view(batch_size, npoint, nsample, -1)
    ray_pos = ray_pos[batch_indices, point_indices].view(batch_size, npoint, nsample, -1)
    return batch_indices, point_indices, q_xyzw, ray_pos


def point_ray_intersection(radius, q_xyz, grouped_vec, grouped_norm):
    q_xyz = torch.unsqueeze(q_xyz, dim=2)
    a_p = grouped_vec - q_xyz
    a_pnn = torch.sum(torch.multiply(a_p, grouped_norm), dim=-1, keepdim=True) * grouped_norm
    distance_v = a_p - a_pnn
    distance = torch.linalg.norm(distance_v, dim=-1)
    valid_mask = torch.lt(distance, radius)
    grouped_ray = grouped_vec - a_pnn

    return grouped_ray, distance_v, valid_mask


def sample_rays(radius, nsample, q_xyz, ray_start, ray_end, m_trans, return_pts=False):
    batch_size = ray_start.shape[0]
    npoint = q_xyz.shape[1]
    indices = sample_from_view(q_xyz, ray_start, m_trans, nsample, return_pts=return_pts)

    xyzw_norm = ray_end[..., :3] - ray_start[..., :3]
    xyzw_norm = xyzw_norm / torch.linalg.norm(xyzw_norm, dim=-1, keepdim=True)

    grouped_vec = ray_start[indices[0], indices[1], :3].reshape(batch_size, npoint, nsample, -1)
    grouped_norm = xyzw_norm[indices[0], indices[1]].reshape(batch_size, npoint, nsample, -1)

    grouped_ray, distance_v, valid_mask = point_ray_intersection(radius, q_xyz, grouped_vec, grouped_norm)
    grouped_ret = torch.cat((grouped_ray, distance_v, grouped_norm), dim=-1)

    if return_pts:
        return grouped_ret, valid_mask, indices

    return grouped_ret, valid_mask


def sample_from_grouped_rays(ray_vec, ray_cnt, nsample=5000):
    x = torch.split(ray_vec, ray_cnt.cpu().numpy().tolist(), dim=0)
    ray_vec = torch.stack([resample_pcd(v, nsample) for v in x], dim=0)
    return ray_vec


def point_ray_vec_dist(ray_vec):
    cur_q_xyz = ray_vec[..., :3] - ray_vec[..., 3:6]
    ray_q_xyz = ray_vec[..., 6:9] - ray_vec[..., 9:12]
    return torch.linalg.norm(cur_q_xyz - ray_q_xyz, dim=-1)


def sample_rays_vec(radius, nsample, q_xyz, partial, pc_end, ray_vec, m_trans):
    batch_size = ray_vec.shape[0]
    npoint = q_xyz.shape[1]
    ray_par = partial.transpose(1, 2)
    ray_par = torch.cat((ray_par, torch.ones(*ray_par.shape[:2], device=partial.device).unsqueeze(-1)), dim=-1)

    ray_ones = torch.ones(*ray_vec.shape[:2], device=ray_vec.device).unsqueeze(-1)
    ray_start = torch.cat((ray_vec[..., :3], ray_ones), dim=-1)
    batch_indices, point_indices = sample_from_view(q_xyz, ray_start, m_trans, nsample)

    ray_ret, ray_mask, indices = sample_rays(radius, nsample, q_xyz, ray_par, pc_end, m_trans, return_pts=True)
    ray_bi, ray_pi, q_pos, ray_pos = indices
    valid_pos = torch.le(q_pos[..., 2], ray_pos[..., 2])
    ray_par = ray_par[ray_bi, ray_pi].reshape(batch_size, npoint, nsample, -1)
    zero_dist = torch.zeros(batch_size, npoint, nsample, 3, device=ray_vec.device)
    ray_ret = torch.cat((ray_ret[..., :-3], ray_par[..., :3], zero_dist, ray_ret[..., -3:]), dim=-1)

    grouped_vec = ray_vec[batch_indices, point_indices].reshape(batch_size, npoint, nsample, -1)
    grouped_start = grouped_vec[..., :3]
    grouped_norm = grouped_vec[..., -3:]
    grouped_ray, distance_v, valid_mask = point_ray_intersection(radius, q_xyz, grouped_start, grouped_norm)
    grouped_ret = torch.cat((grouped_ray, distance_v, grouped_vec), dim=-1)

    ray_distance = point_ray_vec_dist(ray_ret)
    ray_distance[torch.logical_not(valid_pos)] += 1e3
    ray_distance = torch.cat((ray_distance, point_ray_vec_dist(grouped_ret)), dim=2)
    grouped_ret = torch.cat((ray_ret, grouped_ret), dim=2)
    valid_mask = torch.cat((ray_mask, valid_mask), dim=2)
    ray_choice = torch.argsort(ray_distance)[..., :nsample]

    valid_mask = torch.gather(valid_mask, 2, ray_choice)
    ray_choice = ray_choice.unsqueeze(-1).expand(batch_size, npoint, nsample, grouped_ret.shape[-1])
    grouped_ret = torch.gather(grouped_ret, 2, ray_choice)
    return grouped_ret, valid_mask


class RayClsSample(nn.Module):
    def __init__(self):
        super(RayClsSample, self).__init__()

    def forward(self, partial, outs, out1_feat, mean_mst_dis):
        gather1a = torch.cat((outs, out1_feat), dim=1)

        id1a = torch.ones(gather1a.shape[0], 1, gather1a.shape[2]).cuda().contiguous()
        gather1a = torch.cat((gather1a, id1a), 1)
        idpar = torch.zeros(partial.shape[0], 1 + out1_feat.shape[1], partial.shape[2]).cuda().contiguous()
        partial = torch.cat((partial, idpar), 1)
        xx = torch.cat((gather1a, partial), 2)

        resampled_idx = MDS_module.minimum_density_sample(
            xx[:, 0:3, :].transpose(1, 2).contiguous(), outs.shape[2], mean_mst_dis)
        xx = MDS_module.gather_operation(xx, resampled_idx)
        return xx


class WeightNetHidden(nn.Module):
    def __init__(self, input_ch, hidden_units):
        super(WeightNetHidden, self).__init__()
        inout_units = itertools.chain((input_ch,), hidden_units)
        self.hidden_units = torch.nn.ModuleList([torch.nn.Conv2d(in_ch, num_hidden_units, 1)
                                                 for in_ch, num_hidden_units in pairwise(inout_units)])
        self.bn_units = torch.nn.ModuleList([torch.nn.BatchNorm2d(num_hidden_units)
                                             for num_hidden_units in hidden_units])

    def forward(self, xyz):
        net = xyz
        for f_conv, f_bn in zip(self.hidden_units, self.bn_units):
            net = F.relu(f_bn(f_conv(net)))

            # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp{}'.format(i))
        return net
