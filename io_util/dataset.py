import torch.utils.data as data

from .dataflow_client import client_lmdb_dataflow


class ClientDataFlow(data.IterableDataset):
    def __init__(self, **extra_args):
        df_train, num_train = client_lmdb_dataflow(**extra_args)
        self.df_train = df_train
        self.num_train = num_train

    def __len__(self):
        return self.num_train


class ShapeNet(ClientDataFlow):
    def __init__(self, lmdb_train, batchSize, num_points, is_training, **extra_args):
        super(ShapeNet, self).__init__(file=lmdb_train, batch_size=batchSize, num_input_points=5000,
                                       num_gt_points=num_points, is_training=is_training, **extra_args)

    def __iter__(self):
        for cur in self.df_train.get_data():
            feed_dict = {
                'name': tuple(map(str, cur.name)),
                'pc_pl': cur.pc[:, :, :3]
            }

            feed_dict.update({'%s_pl' % s: getattr(cur, s) for s in
                              'ray_start ray_end pc_end m_trans gt_pc'.split()})
            yield feed_dict


class EvalData(ClientDataFlow):
    def __init__(self, lmdb_train, batchSize, num_points, is_training):
        super(EvalData, self).__init__(flow_program='io_util.evalprovider', file=lmdb_train, batch_size=batchSize,
                                       num_input_points=num_points, num_gt_points=num_points, is_training=is_training)

    def __iter__(self):
        for cur in self.df_train.get_data():
            feed_dict = {
                'name': tuple(map(str, cur.name)),
                'pc_pl': cur.pc[..., :3],
                'gt_pc_pl': cur.gt_pc[..., :3]
            }

            yield feed_dict
