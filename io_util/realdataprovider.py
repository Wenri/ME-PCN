import glob
import os
import sys
from collections import namedtuple
from itertools import chain
from time import process_time

import numpy as np
from imageio import imread
from tensorpack.dataflow import DataFromGenerator

from io_util.pc import resample_pcd, resample_pcd_idx
from io_util.provider_base import collect_batch, scan_dir

sem_t = namedtuple(
    typename='local_gen_sem',
    field_names='name pc pc_end ray_start ray_end m_trans gt_pc'
)

load_t = namedtuple(
    typename='local_gen_load',
    field_names='name pts gt_pc m_trans stencil pos'
)

K = np.array([[525, 0, -319.5], [0, 525, -239.5], [0, 0, -1]])


def generate_rays(emptymask, pos_min, pos_max, margin=12):
    rays = np.stack(np.nonzero(emptymask), axis=-1)
    perm = np.logical_and(np.all(rays >= pos_min - margin, axis=-1), np.all(rays <= pos_max + margin, axis=-1))
    return rays[perm]


def inverse_m_trans(ray, m_proj, m_modeview, depth=1.):
    n_pts = ray.shape[0]
    xyzw = depth * np.stack([ray[..., 1], ray[..., 0], np.ones(n_pts)], axis=-1)
    pts = np.linalg.solve(m_proj, xyzw.T)
    xyzw = np.matmul(m_modeview, np.concatenate([pts, np.ones((1, n_pts))], axis=0))
    return xyzw.T


def load_depth_array(m):
    try:
        parent, basename = os.path.split(m)
        category = os.path.basename(parent)
        mask_fname = os.path.join(parent, 'mask.png')
        pts_fname = os.path.join(parent, 'output', 'pts.npy')
        pos_fname = os.path.join(parent, 'output', 'ptsidx.npy')
        m_trans_fname = os.path.join(parent, 'output', 'matrix.npy')
        gt_pc_fname = os.path.join(parent, 'input.npz')
        return load_t(
            name=os.path.join('mask_realdata_test', category, basename), pts=np.load(pts_fname),
            stencil=np.rot90(np.any(imread(mask_fname)[..., :3], axis=-1), k=2),
            pos=np.load(pos_fname), gt_pc=np.load(gt_pc_fname)['points'][..., 2::-1], m_trans=np.load(m_trans_fname)
        )
    except Exception as e:
        print(e, file=sys.stderr)
        sys.stderr.flush()
        return None


def gen_sem(m_files, input_size, output_size, ray_ratio=4):
    while True:
        for f in filter(None, map(load_depth_array, m_files)):
            s = 1.7
            m_scale = np.array([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]])
            m_matrix = np.matmul(m_scale, f.m_trans)

            idx = resample_pcd_idx(f.pts.shape[0], input_size)
            pos_min, pos_max = np.min(f.pos, axis=0), np.max(f.pos, axis=0)
            rays = generate_rays(np.logical_not(f.stencil), pos_min, pos_max)
            rays = resample_pcd(rays, input_size * ray_ratio)

            ray_start = inverse_m_trans(rays, K, m_matrix, 1000)
            ray_end = inverse_m_trans(rays, K, m_matrix, 3000)
            pc_end = inverse_m_trans(f.pos[idx], K, m_matrix, 3000)

            m_modeview = np.linalg.inv(m_matrix)
            m_proj = np.concatenate([K, [[0, 0, -1]]], axis=0)
            m_proj = np.concatenate([m_proj, np.zeros((4, 1))], axis=1)
            yield sem_t(
                name=f.name, pc=s * f.pts[idx, :3], m_trans=np.stack((m_proj.T, m_modeview.T), axis=0),
                gt_pc=resample_pcd(s * f.gt_pc[..., :3], output_size, sort_idx=True),
                ray_start=ray_start, ray_end=ray_end,
                pc_end=pc_end
            )


class RealDataGenerator:
    def __init__(self, lmdb_path, bsize, input_size, output_size):
        self.bsize = bsize
        self.input_size = input_size
        self.output_size = output_size
        self.lmdb_path = lmdb_path
        self.m_files = list(chain.from_iterable(glob.glob(os.path.join(a, 'output')) for a in scan_dir(lmdb_path)))
        self.m_files.sort()

    def __iter__(self):
        gen_func = gen_sem(self.m_files, self.input_size, self.output_size)
        collect_gen = filter(None, gen_func)
        while True:
            start_time = process_time()
            print('loading batch needs {} secs'.format(process_time() - start_time),
                  file=sys.stderr)
            yield collect_batch(collect_gen, self.bsize)

    def __len__(self):
        return len(self.m_files)


def lmdb_dataflow(file, batch_size, num_input_points, num_gt_points, **kwargs):
    gen = RealDataGenerator(file, batch_size, num_input_points, num_gt_points)
    df = DataFromGenerator(gen)
    df.reset_state()

    return df, len(gen), sem_t._fields
