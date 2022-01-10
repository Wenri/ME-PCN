import argparse
import datetime
import json
import os
import random
import sys
from itertools import islice
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from io_util.dataset import ShapeNet
from io_util.vis import GBCVis
from model import MSN
from utils import import_file, AverageValueMeter, weights_init

file_path = os.path.dirname(os.path.realpath(__file__))
emd = import_file('emd_module', os.path.join(file_path, 'emd', 'emd_module.py'))


class FullModel(nn.Module):
    def __init__(self, model):
        super(FullModel, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, inputs, pc_end, ray_start, ray_end, m_trans, gt, eps, iters):
        out = self.model(inputs, pc_end, ray_start, ray_end, m_trans)
        ret = out._asdict()

        dist, _ = self.EMD(out.output1, gt, eps, iters)
        ret['emd1'] = torch.sqrt(dist).mean(1)

        dist, _ = self.EMD(out.output2, gt, eps, iters)
        ret['emd2'] = torch.sqrt(dist).mean(1)

        return ret


class NetworkTrain:
    def __init__(self, network, dataloader, dataloader_val, opt, dir_name=None, lrate=0.001):
        self.best_val_loss = 10
        self.lrate = lrate
        self.dataloader = dataloader
        self.dataloader_val = dataloader_val
        self.network = network
        self.len_dataset = len(dataloader)
        self.len_dataset_val = len(dataloader_val)
        self.auto_save_dir = dir_name

        network.module.model.apply(weights_init)  # initialization of the weight
        if opt.model != '':
            network.module.model.load_state_dict(torch.load(opt.model), strict=False)
            print("Previous weight loaded ")

        self.optimizer = optim.Adam(network.module.model.parameters(), lr=lrate)

        self.train_loss = AverageValueMeter()
        self.val_loss = AverageValueMeter()

        self.train_curve = []
        self.val_curve = []
        self.labels_generated_points = network.module.model.labels_generated_points
        self.env = opt.env
        self.batchSize = opt.batchSize
        self.vis = GBCVis(opt.env)  # set your port

    def valid_epoch(self, epoch, ratio=1.):
        self.val_loss.reset()
        self.network.module.model.eval()
        with torch.no_grad():
            for i, data in enumerate(islice(self.dataloader_val, int(ratio * self.len_dataset_val / self.batchSize))):
                ids, input1, gt = data['name'], data['pc_pl'], data['gt_pc_pl']
                input1 = input1.float().cuda()
                gt = gt.float().cuda()
                input1 = input1.transpose(2, 1).contiguous()
                ray_start, ray_end, m_trans = data['ray_start_pl'], data['ray_end_pl'], data['m_trans_pl']
                ray_start = ray_start.float().cuda()
                ray_end = ray_end.float().cuda()
                m_trans = m_trans.float().cuda()
                pc_end = data['pc_end_pl']
                pc_end = pc_end.float().cuda()
                f = self.network(
                    input1, pc_end, ray_start, ray_end, m_trans, gt=gt.contiguous(), eps=0.005, iters=50
                )
                f = SimpleNamespace(**f)
                self.val_loss.update(f.emd2.mean().item())
                self.vis.scatter_pcd(
                    gt, input1, f.output1, self.labels_generated_points, f.output2, f.ray, f.mask, f.ray_s2, f.mask_s2,
                    prefix='VAL', ids=ids
                )
                print(
                    self.env + ' val [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f' % (
                        epoch, i, int(ratio * self.len_dataset_val / self.batchSize), f.emd1.mean().item(),
                        f.emd2.mean().item(),
                        f.expansion_penalty.mean().item())
                )

    def train_epoch(self, epoch):
        # TRAIN MODE
        self.train_loss.reset()
        self.network.module.model.train()

        # learning rate schedule
        if epoch == 20:
            self.optimizer = optim.Adam(self.network.module.model.parameters(), lr=self.lrate / 10.0)
        if epoch == 40:
            self.optimizer = optim.Adam(self.network.module.model.parameters(), lr=self.lrate / 100.0)

        for i, data in enumerate(islice(self.dataloader, self.len_dataset // self.batchSize)):
            self.optimizer.zero_grad()
            ids, input1, gt = data['name'], data['pc_pl'], data['gt_pc_pl']
            input1 = input1.float().cuda()
            gt = gt.float().cuda()
            input1 = input1.transpose(2, 1).contiguous()
            ray_start, ray_end, m_trans = data['ray_start_pl'], data['ray_end_pl'], data['m_trans_pl']
            ray_start = ray_start.float().cuda()
            ray_end = ray_end.float().cuda()
            m_trans = m_trans.float().cuda()
            pc_end = data['pc_end_pl']
            pc_end = pc_end.float().cuda()
            f = self.network(
                input1, pc_end, ray_start, ray_end, m_trans, gt=gt.contiguous(), eps=0.005, iters=50
            )
            f = SimpleNamespace(**f)
            loss_net = f.emd1.mean() + f.emd2.mean() + f.expansion_penalty.mean() * 0.1
            loss_net.backward()
            self.train_loss.update(f.emd2.mean().item())
            self.optimizer.step()

            if i % 10 == 0:
                self.vis.scatter_pcd(
                    gt, input1, f.output1, self.labels_generated_points, f.output2, f.ray, f.mask, f.ray_s2, f.mask_s2,
                    prefix='TRAIN', ids=ids
                )
            if i and i % 1000 == 0 and self.auto_save_dir is not None:
                self.save(self.auto_save_dir)
                self.valid_epoch(epoch, 1000 * self.batchSize / self.len_dataset)
                self.visualize_loss()

            print(
                self.env + ' train [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f' % (
                    epoch, i, self.len_dataset / self.batchSize, f.emd1.mean().item(), f.emd2.mean().item(),
                    f.expansion_penalty.mean().item())
            )

    def visualize_loss(self):
        self.train_curve.append(self.train_loss.avg)
        self.val_curve.append(self.val_loss.avg)

        self.vis.line(
            X=np.column_stack((np.arange(len(self.train_curve)), np.arange(len(self.val_curve)))),
            Y=np.column_stack((np.array(self.train_curve), np.array(self.val_curve))),
            win='loss',
            opts=dict(title="emd", legend=["train_curve" + self.env, "val_curve" + self.env], markersize=2)
        )
        self.vis.line(
            X=np.column_stack((np.arange(len(self.train_curve)), np.arange(len(self.val_curve)))),
            Y=np.log(np.column_stack((np.array(self.train_curve), np.array(self.val_curve)))),
            win='log',
            opts=dict(title="log_emd", legend=["train_curve" + self.env, "val_curve" + self.env], markersize=2)
        )

    def save(self, dir_name):
        print('saving net...')
        torch.save(self.network.module.model.state_dict(), '%s/network.pth' % dir_name)

    def get_json_stats(self, epoch):
        log_table = {
            "train_loss": self.train_loss.avg,
            "val_loss": self.val_loss.avg,
            "epoch": epoch,
            "lr": self.lrate,
            "bestval": self.best_val_loss,
        }
        return log_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--model', type=str, default='', help='optional reload model path')
    parser.add_argument('--num_points', type=int, default=8192, help='number of points')
    parser.add_argument('--n_primitives', type=int, default=16, help='number of surface elements')
    parser.add_argument('--env', type=str, default="MSN_TRAIN", help='visdom environment')
    parser.add_argument('--lmdb_train', default='./demo/shapenet_renders/')
    parser.add_argument('--log_dir', default='log')

    opt = parser.parse_args()
    print(opt)
    return opt


def main(opt):
    now = datetime.datetime.now()
    save_path = now.isoformat()
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    dir_name = os.path.join(opt.log_dir, save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')
    os.system('cp ./train.py %s' % dir_name)
    os.system('cp ./model.py %s' % dir_name)

    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = ShapeNet(opt.lmdb_train, opt.batchSize, opt.num_points, is_training=True)
    dataloader = data.DataLoader(dataset, batch_size=None)
    dataset_val = ShapeNet(opt.lmdb_train, opt.batchSize, opt.num_points, is_training=True,
                           seed_prefix='render-XYZ.npy')
    dataloader_val = data.DataLoader(dataset_val, batch_size=None)

    print("Train Set Size: {}, Val Set Size: {}".format(len(dataset), len(dataset_val)))

    network = MSN(num_points=opt.num_points, n_primitives=opt.n_primitives, bottleneck_size=2048)
    network = torch.nn.DataParallel(FullModel(network))
    network.cuda()

    trainer = NetworkTrain(network, dataloader, dataloader_val, opt, dir_name=dir_name)

    with open(logname, 'a') as f:  # open and append
        f.write(str(network.module.model) + '\n')

    for epoch in range(opt.nepoch):
        trainer.train_epoch(epoch)
        trainer.save(dir_name)
        if epoch % 1 == 0:  # VALIDATION
            trainer.valid_epoch(epoch)
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(trainer.get_json_stats(epoch)) + '\n')
        trainer.visualize_loss()

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(parse_args()))
