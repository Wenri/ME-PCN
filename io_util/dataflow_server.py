import argparse
import io
import os
import resource
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr, suppress
from distutils.util import strtobool
from importlib import import_module
from pathlib import Path

import numpy as np


def serve_flow_lmdb(flow_program, output, **kwargs):
    try:
        flow_module = import_module(flow_program)
        df, num, fields = flow_module.lmdb_dataflow(**kwargs)
        serve_dataflow(df, num, fields, output)
    except Exception as e:
        output.write('0\n{}: {}\n'.format(type(e).__name__, e).encode())
        print(e, file=sys.stderr)
        raise


def serve_dataflow(df, num, fields, output):
    output.write('{}\n'.format(num).encode())
    output.write('{}\n'.format(' '.join(fields)).encode())
    for d in df.get_data():
        with io.BytesIO() as f:
            np.save(f, d, allow_pickle=True)
            with f.getbuffer() as buf:
                output.write('{}\n'.format(buf.nbytes).encode())
                output.write(buf)


@contextmanager
def setup_fds(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    s_out, s_err = [os.path.join(log_dir, 'std%s.log' % f) for f in ('out', 'err')]
    with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as stdout, \
            open(s_out, 'w') as f_out, open(s_err, 'w') as f_err:
        with redirect_stdout(f_out), redirect_stderr(f_err):
            yield stdout


def setup_fd_limits(max_limit=10240):
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except ValueError:
        no_file = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_limit = min(max_limit, no_file[1])
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, no_file[1]))


def parse_args():
    path = Path(sys.modules[__name__].__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_program', default='io_util.trainvalprovider',
                        help='the py file (default: %(default)s)')
    parser.add_argument('--file', default='data/shapenet/train.lmdb')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=3000)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--is_training', type=strtobool, default=True)
    parser.add_argument('--seed_prefix', type=str, default=None)
    parser.add_argument('--log_dir', default='log/dataflow_server')
    args = parser.parse_args()
    return args


def main(args):
    with suppress(Exception), setup_fds(os.path.join(args.log_dir, os.path.basename(args.file))) as out:
        serve_flow_lmdb(**vars(args), output=out)
    return os.EX_OK


if __name__ == '__main__':
    setup_fd_limits()
    sys.exit(main(parse_args()))
