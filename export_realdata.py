import argparse
import sys
from itertools import islice

import torch.utils.data as data

from io_util.dataset import ShapeNet
from io_util.vis import GBCVis, save_scatter_pcd
from model import *
from utils import *

file_path = os.path.dirname(os.path.realpath(__file__))
emd = import_file('emd_module', os.path.join(file_path, 'emd', 'emd_module.py'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=25, help='input batch size')
    parser.add_argument('--model', type=str, default='./demo/trained_model.pth', help='optional reload model path')
    parser.add_argument('--num_points', type=int, default=8192, help='number of points')
    parser.add_argument('--n_primitives', type=int, default=16, help='number of primitives in the atlas')
    parser.add_argument('--env', type=str, default="MSN_REAL", help='visdom environment')
    parser.add_argument('--lmdb_train', default='./demo/realdata/')

    opt = parser.parse_args()
    print(opt)
    return opt


def main(opt):
    network = MSN(num_points=opt.num_points, n_primitives=opt.n_primitives, bottleneck_size=2048)
    network.cuda()
    network.apply(weights_init)

    if opt.model != '':
        network.load_state_dict(torch.load(opt.model))
        print("Previous weight loaded ")

    network.eval()

    dataset_test = ShapeNet(opt.lmdb_train, opt.batchSize, opt.num_points, is_training=False,
                            flow_program='io_util.realdataprovider')
    dataloader_test = data.DataLoader(dataset_test, batch_size=None)
    len_dataset = len(dataset_test)

    total_batch, rem = divmod(len_dataset, opt.batchSize)
    if rem:
        total_batch += 1

    vis = GBCVis(opt.env)  # set your port

    EMD = emd.emdModule()

    emd_overall = 0
    cd_overall = 0
    with torch.no_grad():
        for i, model in enumerate(islice(dataloader_test, total_batch)):
            ids, partial, gt = model['name'], model['pc_pl'], model['gt_pc_pl']
            ray_start, ray_end, m_trans = model['ray_start_pl'], model['ray_end_pl'], model['m_trans_pl']
            pc_end = model['pc_end_pl']

            if i == total_batch - 1 and rem:
                ids = ids[:rem]
                partial = partial[:rem]
                gt = gt[:rem]
                ray_start = ray_start[:rem]
                ray_end = ray_end[:rem]
                m_trans = m_trans[:rem]
                pc_end = pc_end[:rem]
                opt.batchSize = rem

            partial = partial.float().cuda()
            gt = gt.float().cuda()
            ray_start = ray_start.float().cuda()
            ray_end = ray_end.float().cuda()
            m_trans = m_trans.float().cuda()
            pc_end = pc_end.float().cuda()
            input1 = partial.transpose(2, 1).contiguous()
            output1, output2, ray1, mask1, ray2a, mask2a, ray2b, mask2b, expansion_penalty = network(
                input1, pc_end, ray_start, ray_end, m_trans
            )
            cda, cdb = chemfer_dist(output1, gt)
            cd1 = torch.sqrt(cda) + torch.sqrt(cdb)
            cda, cdb = chemfer_dist(output2, gt)
            cd2 = torch.sqrt(cda) + torch.sqrt(cdb)
            dist, _ = EMD(output1, gt, 0.002, 10000)
            emd1 = torch.sqrt(dist)
            dist, _ = EMD(output2, gt, 0.002, 10000)
            emd2 = torch.sqrt(dist)
            print(opt.env + ' val [%d/%d]  emd1: %f emd2: %f cd1: %f cd2: %f expansion_penalty: %f' % (
                i + 1, total_batch, emd1.mean(), emd2.mean(), cd1.mean(), cd2.mean(), expansion_penalty.mean().item()))
            vis.scatter_pcd(gt, input1, output1, network.labels_generated_points, output2, ray1, mask1, ray2b, mask2b,
                            prefix='TRAIN', ids=ids)
            save_scatter_pcd(gt, partial, output1, network.labels_generated_points, output2, ray1, mask1, ray2a, mask2a,
                             ray2b, mask2b, prefix='export_vis_result', ids=ids)
            cd_overall += cd2.mean()
            emd_overall += emd2.mean()
        print('emd overall: {}'.format(emd_overall / total_batch))
        print('cd overall: {}'.format(cd_overall / total_batch))
    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(parse_args()))
