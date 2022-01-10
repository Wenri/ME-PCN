import glob
import os
import random
import sys
from functools import partial
from itertools import chain, islice
from pathlib import Path
from queue import Queue
from threading import Thread
from time import process_time

import numpy as np

from io_util.rayprocessing import RayProcessingSem


def np_obj_arr(*args):
    arr = np.empty(shape=len(args), dtype=np.object)
    for i, m in enumerate(args):
        arr[i] = np.asanyarray(m)
    return arr


def scan_dir(lmdb_path, sort_as_int=True):
    with os.scandir(lmdb_path) as ds:
        dl = [d for d in ds if d.is_dir()]
    if sort_as_int:
        dl.sort(key=lambda d: int(d.name))
    return dl


class PrepFunctor:
    def __init__(self, lmdb_path, prep):
        self.prep = {k.name: prep(k) for k in lmdb_path}

    def processing(self, nm_args):
        name = Path(nm_args.name).parts[0]
        return self.prep[name].processing(nm_args)


def data_num_gen(dataset_num):
    for dir_list in filter(None, dataset_num):
        dir_num = len(dir_list)
        a, b = divmod(9216, dir_num)
        pick_number = np.full(dir_num, a)
        if a and b:
            pick_plus = np.random.choice(dir_num, b, replace=False)
            pick_number[pick_plus] += 1
        for i, dir_obj in zip(pick_number, dir_list):
            yield i, dir_obj


def collect_batch(collect_gen, bsize):
    return np_obj_arr(*map(np.stack, zip(*islice(collect_gen, bsize))))


class ShapenetGenerator:
    def __init__(self, lmdb_path, bsize, input_size, output_size, is_training, seed_prefix, gen_func):
        self.bsize = bsize
        self.is_training = is_training
        self.gen_func = gen_func
        self.input_size = input_size
        self.output_size = output_size
        if not seed_prefix:
            seed_prefix = 'seed_[0-9A-Z]-XYZ.npy' if self.is_training else 'seed_0-XYZ.npy'
        self.seed_prefix = seed_prefix

        if os.path.isfile(os.path.join(lmdb_path, 'test.txt')):
            self.lmdb_path = lmdb_path
            self.prep = RayProcessingSem
            self.m_files = list(chain.from_iterable(map(self.glob_data_files, self.load_dir(lmdb_path))))
        else:
            self.lmdb_path = scan_dir(lmdb_path, sort_as_int=False)
            self.prep = partial(PrepFunctor, prep=RayProcessingSem)
            self.m_files = list(chain.from_iterable(map(
                self.glob_data_files_num, data_num_gen(self.load_dir(d) for d in self.lmdb_path))))

        if is_training:
            random.seed(0x20211202)
            random.shuffle(self.m_files)
        else:
            self.m_files.sort()

        self.bq = Queue(maxsize=100)

    def load_dir(self, data_dir):
        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            testset = frozenset(filter(None, map(str.strip, f)))

        if not self.is_training:
            return [os.path.join(data_dir, s) for s in testset]

        return [a for a in scan_dir(data_dir) if a.name not in testset]

    def glob_data_files(self, data_dir):
        data_files = glob.iglob(os.path.join(data_dir, self.seed_prefix))
        return data_files

    def glob_data_files_num(self, packed_dir):
        n, data_dir = packed_dir
        data_files = glob.glob(os.path.join(data_dir, self.seed_prefix))
        n = max(1, n)
        random.shuffle(data_files)
        return data_files[:n]

    def __iter__(self):
        gen_thread = Thread(target=self.gen_func, daemon=True,
                            args=(self.m_files, self.bq, self.input_size, self.output_size))
        gen_thread.start()
        processor = self.prep(self.lmdb_path)
        collect_gen = filter(None, map(processor.processing, iter(self.bq.get, None)))
        while True:
            start_time = process_time()
            print('loading batch needs {} secs, queue size {}'.format(process_time() - start_time, self.bq.qsize()),
                  file=sys.stderr)
            yield collect_batch(collect_gen, self.bsize)

    def __len__(self):
        return len(self.m_files)
