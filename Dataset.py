from collections import namedtuple
from tqdm import tqdm
import torch
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from Tools.disps import get_displacements
from Tools.rel_coords import get_relative_coordinates


class SkeletonDataset(Dataset):

    def __init__(self, args, mode):
        super(SkeletonDataset, self).__init__()
        self.list_of_examples = list()
        self.num_classes = args.num_classes
        self.gt_path = args.gt_path
        self.gt_bound_path = args.gt_bound_path
        self.features_path = args.feature_path
        self.sample_rate = args.ds_rate
        self.channel = args.channel
        self.joint_num = args.joint_num
        self.feature_type = args.feature_type
        self.segment_num = args.segment_num
        vid_list_file = args.train_vids_file if mode == 'train' else args.test_vids_file
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        self.dataset_name = args.dataset_name
        self.data = self.load_data()
        file_ptr.close()
        if mode == 'train':
            self.class_weight = self.get_class_weight()
            self.pos_weight = self.get_pos_weight()
        print(f'{mode} dataset built completed')

    def __len__(self):
        return len(self.list_of_examples)

    def __getitem__(self, idx):
        vid = self.list_of_examples[idx]
        feature = self.data[vid]['feature']
        ano = self.data[vid]['ano']
        ano_bound = self.data[vid]['bound']
        feature = torch.from_numpy(feature).float()
        ano = torch.from_numpy(ano)
        ano_bound = torch.from_numpy(ano_bound)

        return feature, ano, ano_bound, vid

    def load_data(self):
        data = {}
        for vid in tqdm(self.list_of_examples):
            ano = np.load(self.gt_path + vid[:-4] + '.npy')
            ano_bound = np.load(self.gt_bound_path + vid[:-4] + '.npy')
            feature = np.load(self.features_path + vid[:-4] + '.npy')
            if self.dataset_name == 'tcg' or self.dataset_name == 'scnu':
                feature = feature[:, :, :, np.newaxis]
                feature = feature.transpose(2, 0, 1, 3)
            elif self.dataset_name == 'pku-mmd':
                feature = feature.reshape(-1, 2, 25, 3).transpose(3, 0, 2, 1)
                # feature = feature.reshape(6, -1, 25, 1)

            feature = self.get_features(self.feature_type, feature, root_node=0)

            if self.dataset_name == 'pku-mmd':
                feature = feature.transpose(3, 0, 1, 2).reshape(12, -1, 25, 1)


            if self.sample_rate != 1:
                feature = feature[:, ::self.sample_rate, :, :]  # temporal sampling
                ano = ano[::self.sample_rate]
                # not to remove boundary at odd number frames.
                idx = np.where(ano_bound == 1.0)[0]
                ano_bound = np.zeros(len(ano))
                ano_bound[np.floor_divide(idx, self.sample_rate).astype(int)] = 1.0

            data[vid] = {'feature': feature, 'ano': ano, 'bound': ano_bound}
        return data

    def collate_fn(self, batch_data):
        batch_input = []
        batch_target = []
        batch_boundary = []
        batch_name = []
        for d in batch_data:
            batch_input.append(d[0])
            batch_target.append(d[1])
            batch_boundary.append(d[2])
            batch_name.append(d[3])

        length_of_sequences = list(map(len, batch_target))
        max_length = (max(length_of_sequences)//self.segment_num+1)*self.segment_num
        batch_input_tensor = torch.zeros(len(batch_input), self.channel, max_length, self.joint_num, 1,
                                         dtype=torch.float) #tensor（batch，channel，Temporal_max，node_num,1）
        batch_target_tensor = torch.ones(len(batch_input), max_length, dtype=torch.long) * (-100) #-100tensor （batch,Temporal_max)
        batch_boundary_tensor = torch.ones(len(batch_input), max_length, dtype=torch.long) * (0.)  #-100tensor （batch,Temporal_max)
        mask = torch.zeros(len(batch_input), self.num_classes, max_length, dtype=torch.float) #mask 0 （batch，class，temporal）
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1], :, :] = batch_input[i]
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = batch_target[i]
            batch_boundary_tensor[i, :np.shape(batch_boundary[i])[0]] = batch_boundary[i]
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, batch_target[i].shape[0])

        return {
            'feature': batch_input_tensor,
            'annotation': batch_target_tensor,
            "boundary": batch_boundary_tensor.to(torch.float32),
            'mask': mask,
            'names': batch_name
        }
    def get_class_weight(self) -> torch.Tensor:
        nums = [0 for i in range(self.num_classes)]
        for v_data in self.data:
            v_ano = self.data[v_data]['ano']
            num, cnt = np.unique(v_ano, return_counts=True)
            for n, c in zip(num, cnt):
                nums[int(n)] += int(c)
        class_num = torch.tensor(nums)
        total = class_num.sum().item()
        frequency = class_num.float() / total
        median = torch.median(frequency)
        class_weight = median / frequency

        return class_weight


    def get_pos_weight(self) -> torch.Tensor:
        """
        pos_weight for binary cross entropy with logits loss
        pos_weight is defined as reciprocal of ratio of positive samples in the dataset
        """

        n_classes = 2  # boundary or not 二分类
        nums = [0 for cls in range(n_classes)]  # 初始化一个长度为类别总数的列表，用于存储每个类别的样本数量
        for v_data in self.data:  # 循环遍历data中每一个特征序列
            v_bound = self.data[v_data]['bound']  # 拿到标签
            num, cnt = np.unique(v_bound, return_counts=True)  # 统计标签中每个类别的样本数量
            for n, c in zip(num, cnt):
                nums[int(n)] += c  # 将每个类别的样本数量累加到相应的位置

            pos_ratio = nums[1] / sum(nums)
            pos_weight = 1 / pos_ratio


        return torch.tensor(pos_weight)  # 权重=总数/边界数

    @staticmethod
    def get_features(type, sample, root_node):
        if type == 'new':
            disps = get_displacements(sample)
            rel_coords = get_relative_coordinates(sample, references=(root_node))
            sample = np.concatenate([disps, rel_coords], axis=0)
        return sample


if __name__ == '__main__':
    args = {
        'ds_rate': 2,
        'dataset_name': 'lara',
        'train_vids_file': '/share/Datasets/lara/splits_loso_validation/train.split1.bundle',
        'test_vids_file': '/share/Datasets/lara/splits_loso_validation/test.split1.bundle',
        'channel': 6,
        'num_classes': 8,
        'gt_path': '/share/Datasets/lara/groundTruth_/',
        'feature_path': '/share/Datasets/lara/features7/',
        'joint_num': 19,
        'feature_type': 'origin',
    }
    Args = namedtuple('ArgTuple', args.keys())
    dataset = SkeletonDataset(Args(**args), mode='test')
    data = dataset.data
    r = dict()
    max = 0
    min = 100000
    l = []
    for vid in data:
        ano = data[vid]['ano']
        seg_idx = (ano[1:] - ano[:-1]).nonzero()[0]
        seg_cls = ano[seg_idx]
        for i, (idx, cls) in enumerate(zip(seg_idx, seg_cls)):
            cls = str(cls)
            if not r.__contains__(cls):
                r[cls] = []
            if i == 0:
                r[cls].append(idx)
            elif i == (len(seg_idx) - 1):
                r[cls].append(len(ano) - idx)
            else:
                r[cls].append(seg_idx[i] - seg_idx[i - 1])
    print('')
