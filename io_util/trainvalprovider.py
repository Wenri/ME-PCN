import sys
from collections import namedtuple

import numpy as np
from tensorpack.dataflow import DataFromGenerator

from io_util.pc import resample_pcd
from io_util.rayprocessing import RayProcessingSem
from io_util.provider_base import ShapenetGenerator
from io_util.provider_fileio import load_depth_array

sem_t = namedtuple(
    typename='local_gen_sem',
    field_names='name pts rays m_trans gt_pc'
)


def invert_viewport_trans(x, y, shape):
    w, h = shape
    x = (x.astype(np.float32) + 0.5) * (2.0 / w) - 1.0
    y = (y.astype(np.float32) + 0.5) * (2.0 / h) - 1.0
    return x, y


def generate_rays(emptymask, pos_min, pos_max, margin=0.01):
    rays = np.stack(invert_viewport_trans(*np.nonzero(emptymask), emptymask.shape), axis=-1)
    perm = np.logical_and(np.all(rays >= pos_min - margin, axis=-1), np.all(rays <= pos_max + margin, axis=-1))
    return rays[perm]


def gen_sem(m_files, bq, input_size, output_size, ray_ratio=4):
    while True:
        for f in filter(None, map(load_depth_array, m_files)):
            try:
                pos = np.stack(invert_viewport_trans(*np.nonzero(f.stencil), f.stencil.shape), axis=-1)
                pos_min, pos_max = np.min(pos, axis=0), np.max(pos, axis=0)
                rays = generate_rays(np.logical_not(f.stencil), pos_min, pos_max)
                bq.put(sem_t(
                    name=f.name, pts=resample_pcd(f.pts, input_size, sort_idx=True),
                    rays=resample_pcd(rays, input_size * ray_ratio), m_trans=np.array(f.m_trans),
                    gt_pc=resample_pcd(f.gt_pc, output_size, sort_idx=True),
                ))
            except Exception as e:
                print(f.name, e, file=sys.stderr)


def lmdb_dataflow(file, batch_size, num_input_points, num_gt_points, is_training, seed_prefix, **kwargs):
    gen = ShapenetGenerator(file, batch_size, num_input_points, num_gt_points, is_training, seed_prefix,
                            gen_func=gen_sem)
    df = DataFromGenerator(gen)
    df.reset_state()

    return df, len(gen), RayProcessingSem.ins_t._fields
