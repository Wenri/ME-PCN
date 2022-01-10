import os
import random
import struct
from contextlib import suppress
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import visdom


def write_pointcloud(filename, xyz_points, rgb_points=None):
    """ creates a .ply file of the point clouds generated
    """

    n_total, n_dim = xyz_points.shape
    assert n_dim == 3, 'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.asarray((255, 0, 0), dtype=np.uint8)
    if rgb_points.ndim < 2:
        rgb_points = np.broadcast_to(np.expand_dims(rgb_points, axis=0), shape=(n_total, 3))
    assert xyz_points.shape == rgb_points.shape, 'Input RGB colors should be Nx3 float array and have same size as ' \
                                                 'input XYZ points '

    # Write header of .ply file
    with open(filename, 'wb') as fid:
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(xyz_points.shape[0]):
            fid.write(bytearray(struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                            rgb_points[i, 0].tostring(), rgb_points[i, 1].tostring(),
                                            rgb_points[i, 2].tostring())))


def save_pcn_result(out_dir, visu_batch, **matcontent):
    pcn_dir = os.path.join(out_dir, 'pcn')
    if visu_batch == 0:
        os.makedirs(pcn_dir, exist_ok=True)
    file_name = os.path.join(pcn_dir, 'matcontent_{}.mat'.format(visu_batch))
    hdf5storage.write(matcontent, filename=file_name, matlab_compatible=True, truncate_existing=True)


def export_model_result(out_dir, ids, pts, coarse, gts):
    data_dir = os.path.join(out_dir, 'exports')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    for name, pc, coarse_pc, gt in zip(ids, pts, coarse, gts):
        category, name = name.split('/')[:2]
        save_dir = os.path.join(data_dir, category.split('_')[1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, '{}_pts.npy'.format(name)), pc)
        np.save(os.path.join(save_dir, '{}_coarse.npy'.format(name)), coarse_pc)
        np.save(os.path.join(save_dir, '{}_gt.npy'.format(name)), gt)


def save_scatter_pcd(gt, input1, output1, labels_generated_points, output2, ray1, mask1, ray2a, mask2a, ray2b, mask2b,
                     prefix, ids):
    for idx, idstr in enumerate(ids):
        category, name, seed = idstr.split('/')
        save_dir = Path(prefix, category.split('_')[1], name, seed)

        if not save_dir.exists():
            os.makedirs(save_dir, exist_ok=True)

        input_pts = input1[idx]
        ray_pts = ray1[idx, ..., :3].data[mask1[idx]]
        ray1_pts = ray2a[idx, ..., :3].data[mask2a[idx]]
        ray2_pts = ray2b[idx, ..., 6:9].data[mask2b[idx]]
        ray2_label = torch.count_nonzero(ray2b[idx, ..., 9:12].data[mask2b[idx]], dim=-1).bool().int() + 1

        write_pointcloud(save_dir / 'gt.ply', gt[idx].data.cpu()[:, :3])
        write_pointcloud(save_dir / 'input_pts.ply', input_pts.cpu())
        write_pointcloud(save_dir / 'input_rays.ply', ray_pts.cpu())
        write_pointcloud(save_dir / 'output_coarse.ply', output1[idx].cpu())
        np.save(save_dir / 'output_coarse_label.npy', labels_generated_points[0:output1.size(1)])
        write_pointcloud(save_dir / 'ray1_pts.ply', ray1_pts.cpu())
        write_pointcloud(save_dir / 'ray2_pts.ply', ray2_pts.cpu())
        np.save(save_dir / 'ray2_label', ray2_label.cpu())
        write_pointcloud(save_dir / 'output_fine.ply', output2[idx].cpu())


class GBCVis(visdom.Visdom):
    def __init__(self, env):
        try:
            super(GBCVis, self).__init__(port=8097, env=env, raise_exceptions=True)
        except ConnectionError:
            super(GBCVis, self).__init__(log_to_filename=os.devnull, env=env, offline=True)

    def scatter(*args, **kwargs):
        with suppress(Exception):
            return super(GBCVis, *args, **kwargs).scatter()

    def scatter_pcd(self, gt, input1, output1, labels_generated_points, output2, ray1, mask1, ray2, mask2,
                    prefix, ids):
        idx = random.randint(0, input1.size()[0] - 1)
        input_pts = input1.transpose(2, 1)[idx]
        ray_pts = ray1[idx, ..., :3].data[mask1[idx]]
        ray2_pts = ray2[idx, ..., 6:9].data[mask2[idx]]
        ray2_label = torch.count_nonzero(ray2[idx, ..., 9:12].data[mask2[idx]], dim=-1).bool().int() + 1
        labels_input = torch.ones(input_pts.size()[0], dtype=torch.int, device=input_pts.device)
        labels_ray = torch.full((ray_pts.size()[0],), 2, dtype=torch.int, device=input_pts.device)
        self.scatter(X=gt[idx].data.cpu()[:, :3],
                     win=prefix + '_GT',
                     opts=dict(title=prefix + '_GT_' + ids[idx], markersize=2),
                     )
        self.scatter(X=torch.cat((input_pts, ray_pts)).cpu(),
                     Y=torch.cat((labels_input, labels_ray)).cpu(),
                     win=prefix + '_INPUT',
                     opts=dict(title=prefix + '_INPUT_' + ids[idx], markersize=2),
                     )
        self.scatter(X=output1[idx].cpu(),
                     Y=labels_generated_points[0:output1.size(1)],
                     win=prefix + '_COARSE',
                     opts=dict(title=prefix + '_COARSE_' + ids[idx], markersize=2),
                     )
        self.scatter(X=ray2_pts.cpu(),
                     Y=ray2_label.cpu(),
                     win=prefix + '_CORSERAY',
                     opts=dict(title=prefix + '_CORSERAY_' + ids[idx], markersize=2),
                     )
        self.scatter(X=output2[idx].cpu(),
                     win=prefix + '_OUTPUT',
                     opts=dict(title=prefix + '_OUTPUT_' + ids[idx], markersize=2),
                     )
        return idx
