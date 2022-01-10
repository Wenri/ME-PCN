# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
import io
import shlex
import subprocess
import sys
from collections import namedtuple
from contextlib import closing
from functools import partial

import numpy as np
from tensorpack.dataflow import DataFlow, MultiThreadRunner


class ClientGenerator(DataFlow):
    def __init__(self, client_args, retry=True):
        print("Shell: " + ' '.join(shlex.quote(a) for a in client_args))
        self.client_args = client_args
        self.proc = None
        self._len = 0
        self.batch_t = None
        self.retry = retry
        self.start_or_restart()

    def start_or_restart(self):
        self.proc = subprocess.Popen(self.client_args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, close_fds=True)
        num = int(self.proc.stdout.readline().decode())
        msg = self.proc.stdout.readline().decode()
        if not num:
            raise ValueError(msg)
        self.batch_t = namedtuple(typename='cli_gen_batch_t', field_names=msg.split())
        if self._len and self._len != num:
            print("Warning: num {} != new_num {}".format(self._len, num), file=sys.stderr)
        self._len = num

    def decode_one(self):
        n_bytes = int(self.proc.stdout.readline().decode())
        if not n_bytes:
            msg = self.proc.stdout.readline().decode()
            raise ValueError(msg)
        buf = self.proc.stdout.read(n_bytes)
        assert len(buf) == n_bytes, "expected to read {} bytes, but got {}".format(n_bytes, len(buf))
        with io.BytesIO(buf) as f:
            return self.batch_t._make(np.load(f, allow_pickle=True))

    def __iter__(self):
        while True:
            if self.proc.poll() is None:
                try:
                    with self.proc:
                        while self.proc.stdout:
                            yield self.decode_one()
                    return
                except Exception as e:
                    print("client_generator_error: {}".format(e), file=sys.stderr)
                    if not self.retry:
                        raise
            elif self.proc.returncode != 0:
                print("Pervious generator exited with code: {}".format(self.proc.returncode), file=sys.stderr)
                if not self.retry:
                    print("Not retry due to non-zero exit code.")
                    return
            self.start_or_restart()

    def __len__(self):
        return self._len


class MultiThreadWithClosing(MultiThreadRunner):
    class _Worker(MultiThreadRunner._Worker):
        def run(self):
            self.df.reset_state()
            try:
                while True:
                    with closing(self.df.get_data()) as ds:
                        for dp in ds:
                            self.queue_put_stoppable(self.queue, dp)
                            if self.stopped():
                                return
            except Exception:
                if not self.stopped():  # skip duplicated error messages
                    raise
            finally:
                self.stop()

    def __init__(self, get_df, num_prefetch, num_thread):
        super().__init__(get_df, num_prefetch, num_thread)
        self.threads = [self._Worker(lambda: t.df, self.queue) for t in self.threads]

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, 'threads'):
            super().__del__()


def client_lmdb_dataflow(**kwargs):
    """
     (db_path, batch_size, input_size, output_size, is_training, test_speed=False)
    """
    cli_args = [sys.executable, '-m', 'io_util.dataflow_server']
    cli_args.extend('--{}={}'.format(k, v) for k, v in kwargs.items())

    df = MultiThreadWithClosing(partial(ClientGenerator, cli_args), num_prefetch=50, num_thread=1)
    df.reset_state()

    return df, len(df)
