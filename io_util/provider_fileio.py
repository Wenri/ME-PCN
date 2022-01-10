import os
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
from imageio import imread
from numpy.lib.format import open_memmap

load_t = namedtuple(
    typename='local_gen_load',
    field_names='name pts gt_pc m_trans stencil'
)


def load_depth_array(m):
    try:
        parts = Path(m).parts
        basename = parts[-1][:-8]
        for prefix in ('NOISE2_', 'NOISE_', ''):
            stencil_fname = os.path.join(*parts[:-1], prefix + basename + '-STENCIL.png')
            if os.path.exists(stencil_fname):
                break
        m_trans_fname = os.path.join(*parts[:-1], basename + '-MATRIX.npy')
        gt_pc_fname = os.path.join(*parts[:-1], 'render-GT_PC.npy')
        return load_t(
            name=os.path.join(*parts[-3:-1], basename), pts=open_memmap(m, mode='r'),
            gt_pc=open_memmap(gt_pc_fname, mode='r'), m_trans=open_memmap(m_trans_fname, mode='r'),
            stencil=np.flipud(imread(stencil_fname))
        )
    except Exception as e:
        print(e, file=sys.stderr)
        return None
