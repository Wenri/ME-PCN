import sys
from collections import namedtuple

import numpy as np


def np_obj(m):
    arr = np.empty(shape=1, dtype=np.object)
    arr[0] = np.asanyarray(m)
    arr.shape = ()
    return arr


def inverse_m_trans(ray, m_proj, m_modeview, depth=1.):
    assert np.allclose(m_proj[:2, 2:], 0) and np.allclose(m_proj[2:, -1], [-1.0, 0.0])

    m_a, m_b = m_proj[2:, 2]
    neg_z = m_b / (depth + m_a)
    ray = ray.T * neg_z
    depth = np.repeat(depth * neg_z, ray.shape[-1])
    neg_z = np.repeat(neg_z, ray.shape[-1])

    xyzw = np.concatenate([np.flipud(ray), np.expand_dims(depth, axis=0), np.expand_dims(neg_z, axis=0)], axis=0)
    xyzw = np.linalg.solve(np.matmul(m_modeview, m_proj).T, xyzw)

    return xyzw.T


def pts_m_trans(pts, m_proj, m_modeview, depth=1.):
    ray = np.concatenate((pts[..., :3], np.expand_dims(np.ones(pts.shape[0]), axis=-1)), axis=-1)
    ray = np.matmul(ray, np.matmul(m_modeview, m_proj))
    ray /= np.expand_dims(ray[..., 3], axis=-1)
    return inverse_m_trans(ray[..., 1::-1], m_proj, m_modeview, depth)


class RayProcessingSem:
    ins_t = namedtuple(
        typename='PreprocessingIns_t',
        field_names='name pc pc_end ray_start ray_end m_trans gt_pc'
    )

    def __init__(self, db_path):
        self.db_path = db_path

    def processing(self, nm_args):
        try:
            return self.ins_t(
                pc=nm_args.pts, m_trans=nm_args.m_trans,
                gt_pc=nm_args.gt_pc, name=np_obj(nm_args.name), pc_end=pts_m_trans(nm_args.pts, *nm_args.m_trans),
                ray_start=inverse_m_trans(nm_args.rays, *nm_args.m_trans, depth=-1.),
                ray_end=inverse_m_trans(nm_args.rays, *nm_args.m_trans),
            )
        except Exception as e:
            print('Skipping {} due to {}'.format(nm_args.name, e), file=sys.stderr)
            return None
