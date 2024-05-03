# region
import os
import warnings
import torch.distributed as dist

import torchviz

from Tools.SelectCUDA import get_cuda_id, load_yaml
import torch

import sys
from importlib import import_module
import numpy as np
import random
import matplotlib.pyplot as plt

from Tools.metric import ScoreMeter
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import SkeletonDataset

import argparse

parser = argparse.ArgumentParser()
# general arguments
parser.add_argument('--dataset', default='hugadb', type=str, help='dataset root path')
parser.add_argument('--netname', default='ME-ST', type=str, help='MAE-ST or MSGCN')
parser.add_argument('--cuda', default=0, type=int, help='cuda id')
parser.add_argument('--model_path', default=None, type=str, help='MAE-ST or MSGCN')
parser.add_argument('--seed', default=0, type=int, help='seed')

Args = parser.parse_args()

torch.manual_seed(Args.seed)
np.random.seed(Args.seed)
random.seed(Args.seed)
torch.cuda.manual_seed(Args.seed)
torch.cuda.manual_seed_all(Args.seed)
torch.backends.cudnn.benchmark = True

# cuda_id = get_cuda_id()
cuda_id = Args.cuda
print('run on cuda:', cuda_id)
torch.cuda.set_device(cuda_id)
os.environ["CUDA_VISIBLE_DEVICES"] = f'{cuda_id}'

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"
# endregion
user_name='result'
dataset = Args.dataset
args, args_pretty = load_yaml(f'configs/{dataset}.yaml')
net_name = Args.netname
method = f'test'
batch_size = args.batch_size

if Args.model_path is not None:
    trained_model = Args.model_path
else:
    trained_model = f'pre_trained_models/ME-ST/{dataset}/best_model.pt'
# region
learning_rate = args.learning_rate
if not os.path.exists(f'{user_name}/{net_name}/{dataset}/{method}'):
    os.makedirs(f'{user_name}/{net_name}/{dataset}/{method}')

testset = SkeletonDataset(args, mode='test')
test_dataloader = DataLoader(dataset=testset, batch_size= 2*batch_size,
                             collate_fn=testset.collate_fn,
                             shuffle=False, num_workers=8, drop_last=False)
ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
mse = nn.MSELoss(reduction='none')
# endregion
Model = getattr(import_module(f'models.{net_name}'),'Model') # from models.MSGCN import Model
net = Model(args).cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.to(device)
net.load_state_dict(torch.load(trained_model,map_location='cuda:0'))
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) #

print("use ReduceLSOnPlateau scheduler")


def eval():
    net.eval()
    evaled = []
    scores = []
    n_results = 5
    for i in range(n_results):
        scores.append(ScoreMeter(
            iou_thresholds=(0.1, 0.25, 0.5),
            n_classes=args.num_classes
        ))
    with torch.no_grad():
        index_data=0


        for data in tqdm(test_dataloader):
            index_data+=1
            x = data['feature'].to(device)
            t = data['annotation'].to(device)
            mask = data['mask'].to(device)
            names = data['names']
            predictions = net(x, mask)
            batch_size = x.shape[0]
            for i in range(batch_size):
                name = names[i]
                if evaled.__contains__(name):
                    continue
                else:
                    evaled.append(name)
                    tobeshown = []
                    for idx, prob in enumerate([prediction_item[i] for prediction_item in predictions]):
                        prob = prob.detach()
                        _, predicted = torch.max(prob, 0)
                        target = t[i].cpu().numpy()
                        m = mask[i]
                        cnt = int(torch.sum(m[0]))
                        prob = prob[:, :cnt]
                        target = target[:cnt]
                        prediction = torch.nn.Softmax(dim=0)(prob)
                        predicted = torch.max(prediction, dim=0)[1]
                        predicted = F.interpolate(predicted.unsqueeze(0).unsqueeze(0).float(),
                                                  size=target.shape[0]).squeeze().long()
                        predicted = predicted.cpu().data.numpy()

                        scores[idx].update(predicted, target)
                        tobeshown.append(predicted)

                        if i==0 and index_data == 1:
                            if idx==0:
                                targets.append(target)
                                epoch_targets.append(1)
                            predicts[0].append(predicted)


        _scores = [score.get_scores() for score in scores]

        test_accuracies.append(round(_scores[4][0], 3))
        test_editscores.append(round(_scores[4][1], 3))
        test_f1scores.append(round(_scores[4][2][2], 3))

        result_dicts = [
            {'acc': f'{round(_scores[i][0], 3)}',
             'edit': f'{round(_scores[i][1], 3)}',
             'f1': f'{round(_scores[i][2][0], 3)}',
             'f2': f'{round(_scores[i][2][1], 3)}',
             'f3': f'{round(_scores[i][2][2], 3)}',
             }
            for i in range(len(scores))
        ]
        for idx, result_dict in enumerate(result_dicts):
            print(f'stage{idx}:', result_dict)

        with open(f'{user_name}/{net_name}/{dataset}/{method}/scores_{net_name}.txt', "a+") as file:
            for idx, result_dict in enumerate(result_dicts):
                file.write(f'stage{idx}:{result_dict}\n')
            file.write('\n------------------------------------\n')



# region
results = []
result_avg = {'acc': 0.0,
             'edit': 0.0,
             'f1': 0.0,
             'f2': 0.0,
             'f3': 0.0,
             }
train_losses = []
train_accuracies = []
test_accuracies = []
test_editscores = []
test_f1scores = []
epoch_targets=[]
targets = []
predicts = [[] for _ in range(args.num_stages+1)]
with open(f'{user_name}/{net_name}/{dataset}/{method}/scores_{net_name}.txt', "w") as file:
    file.write(f'The result printed:{method}')


    result = eval()
    results.append(result)


