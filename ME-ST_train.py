# region
import os
import warnings

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
method = f'train'
batch_size = args.batch_size
# region
learning_rate = args.learning_rate
if not os.path.exists(f'{user_name}/{net_name}/{dataset}/{method}'):
    os.makedirs(f'{user_name}/{net_name}/{dataset}/{method}')

trainset = SkeletonDataset(args, mode='train')
train_dataloader = DataLoader(dataset=trainset, batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              shuffle=True, num_workers=8, drop_last=False)
testset = SkeletonDataset(args, mode='test')
test_dataloader = DataLoader(dataset=testset, batch_size= 2*batch_size,
                             collate_fn=testset.collate_fn,
                             shuffle=False, num_workers=8, drop_last=False)
ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
mse = nn.MSELoss(reduction='none')
# endregion
Model = getattr(import_module(f'models.{net_name}'),'Model') # from models.MSGCN import Model
net = Model(args).cuda()

device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) #

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8,
#                                          verbose=True)
# scheduler_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, mode="min",factor=0.8,
#                                                         verbose=True)
print("use ReduceLSOnPlateau scheduler")


def train():
    global train_accuracies, train_losses
    net.train()
    net.to(device)
    n_results = args.num_stages+1
    epoch_stage_losses = [0] * n_results
    totals = [0] * n_results
    tps = [0] * n_results
    for data in tqdm(train_dataloader, file=sys.stdout):
        features = data['feature'].to(device)
        label = data['annotation'].to(device)
        mask = data['mask'].to(device)
        # print("device:", next(net.parameters()).device)
        # print("device:",features.device,label.device,mask.device)
        optimizer.zero_grad()
        output_list = net(features, mask)
        loss = 0
        for idx, prob in enumerate(output_list):
            # prob (batch_size, num_classes, sequence_length)
            _loss = 0.15 * torch.mean(torch.clamp(
                mse(F.log_softmax(prob[:, :, 1:], dim=1), F.log_softmax(prob.detach()[:, :, :-1], dim=1)), min=0,
                max=16) * mask[:, :, 1:])

            prob = prob.transpose(2, 1).contiguous() #prob （batch,temporal,class)
            _loss += ce(prob.view(-1, args.num_classes), label.view(-1))
            loss += _loss
            prob= prob.detach()
            _, predicted = torch.max(prob, 2)
            totals[idx] += torch.sum(mask[:, 0, :]).item()
            tps[idx] += ((predicted == label).float() * mask[:, 0, :].squeeze(1)).sum().item()
            epoch_stage_losses[idx] += _loss.detach().item()

        loss.backward()
        # optimizer.module.step()
        optimizer.step()

    # scheduler.step()
    # scheduler_reduce.step(epoch_stage_losses[4])

    train_accuracies.append(round(np.mean(tps[args.num_stages] / totals[args.num_stages]), 3))
    train_losses.append(round(epoch_stage_losses[args.num_stages] / len(train_dataloader), 3))

    for idx, (total, tp, loss) in enumerate(zip(totals, tps, epoch_stage_losses)):
        print(f'stage {idx} : acc={round(np.mean(tp / total), 3)}  loss={round(loss / len(train_dataloader), 3)}')

    with open(f'{user_name}/{net_name}/{dataset}/{method}/scores_{net_name}.txt', "a+") as file:
        file.write(f'epoch+1:{e}')
        for idx, (total, tp, loss) in enumerate(zip(totals, tps, epoch_stage_losses)):
            file.write(f'stage {idx} : acc={round(np.mean(tp / total), 3)}  loss={round(loss / len(train_dataloader), 3)}\n')


def eval():
    global best_acc, best_edit, best_f1
    global test_accuracies, test_editscores, test_f1scores
    net.eval()
    evaled = []
    scores = []
    n_results = args.num_stages+1
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

                        _, predicted = torch.max(prob, 0)
                        target = t[i].cpu().numpy()
                        # print("target:",target.size,target)
                        m = mask[i]
                        cnt = int(torch.sum(m[0]))
                        prob = prob[:, :cnt]
                        target = target[:cnt]
                        prediction = torch.nn.Softmax(dim=0)(prob) #softmax

                        predicted = torch.max(prediction, dim=0)[1]

                        predicted = F.interpolate(predicted.unsqueeze(0).unsqueeze(0).float(),
                                                  size=target.shape[0]).squeeze().long()

                        predicted = predicted.cpu().data.numpy()

                        scores[idx].update(predicted, target)
                        tobeshown.append(predicted)

                        # if e % 50 ==0 and i==0 and index_data == 1:
                        #     if idx==0:
                        #         targets.append(target)
                        #         epoch_targets.append(e)
                        #     predicts[int(e/50-1)].append(predicted)


        _scores = [score.get_scores() for score in scores]

        test_accuracies.append(round(_scores[args.num_stages][0], 3))
        test_editscores.append(round(_scores[args.num_stages][1], 3))
        test_f1scores.append(round(_scores[args.num_stages][2][2], 3))

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

        if float(best_acc['acc']) < float(result_dicts[-1]['acc']):
            best_acc['epoch'] = e
            for key in result_dicts[-1]:
                best_acc[key] = result_dicts[-1][key]
            torch.save(net.state_dict(), f'{user_name}/{net_name}/{dataset}/{method}/best_acc_model.pt')
        if float(best_edit['edit']) < float(result_dicts[-1]['edit']):
            best_edit['epoch'] = e
            for key in result_dicts[-1]:
                best_edit[key] = result_dicts[-1][key]
            torch.save(net.state_dict(), f'{user_name}/{net_name}/{dataset}/{method}/best_edit_model.pt')
        if float(best_f1['f3']) < float(result_dicts[-1]['f3']):
            best_f1['epoch'] = e
            for key in result_dicts[-1]:
                best_f1[key] = result_dicts[-1][key]
            torch.save(net.state_dict(), f'{user_name}/{net_name}/{dataset}/{method}/best_f1_model.pt')


        return result_dicts[-1]


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
best_acc = { 'epoch': 0,
             'acc': '0.000',
             'edit': '0.000',
             'f1':'0.000',
             'f2': '0.000',
             'f3': '0.000',}
best_edit = { 'epoch': 0,
             'acc': '0.000',
             'edit': '0.000',
             'f1':'0.000',
             'f2': '0.000',
             'f3': '0.000',}
best_f1 = { 'epoch': 0,
             'acc': '0.000',
             'edit': '0.000',
             'f1':'0.000',
             'f2': '0.000',
             'f3': '0.000',}
epoch_targets=[]
targets = []
predicts = [[] for _ in range(3)]
with open(f'{user_name}/{net_name}/{dataset}/{method}/scores_{net_name}.txt', "w") as file:
    file.write(f'The result printed:{method}')

for epoch in range(args.num_epochs):
    print('epoch', epoch, ':')

    e = epoch + 1

    train()

    if e % args.period_epoch == 0 and e > 0:
        result = eval()
        results.append(result)

#train fig
plt.figure()
plt.plot(range(1,args.num_epochs+1),train_losses,label="Training Loss")
plt.plot(range(1,args.num_epochs+1),train_accuracies,label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Train Value")
plt.legend()
plt.title("Training loss and accuracy")
plt.savefig(f'{user_name}/{net_name}/{dataset}/{method}/{net_name}_train_plt.png')

#test fig
plt.figure()
plt.plot(range(1,args.num_epochs+1,args.period_epoch),test_accuracies,label="Test Accuracy")
plt.plot(range(1,args.num_epochs+1,args.period_epoch),test_editscores,label="Test Edit Score")
plt.plot(range(1,args.num_epochs+1,args.period_epoch),test_f1scores,label="Test F1@50 Score")
plt.xlabel("Epoch")
plt.ylabel("Test Value")
plt.legend()
plt.title("Test accuracy edit f1 score")
plt.savefig(f'{user_name}/{net_name}/{dataset}/{method}/{net_name}_test_plt.png')

# plt.show()


for idx, result in enumerate(results):
    print(f'evaluation {idx+1}:',result)

with open(f'{user_name}/{net_name}/{dataset}/{method}/scores_{net_name}.txt', "a+") as file:
    for idx, result in enumerate(results):
        file.write(f'evaluation {idx+1}:{result}\n')

for idx in range(100, len(results)):
    for key, value in results[idx].items():
        result_avg[key] += (float(value)/(len(results)-100))

print('Avg_Result:', result_avg)
with open(f'{user_name}/{net_name}/{dataset}/{method}/scores_{net_name}.txt', "a+") as file:
    file.write(f'Avg_Result: {result_avg}\n')

print('Best_acc_Result:', best_acc)
print('Best_edit_Result:', best_edit)
print('Best_f1_Result:', best_f1)
with open(f'{user_name}/{net_name}/{dataset}/{method}/scores_{net_name}.txt', "a+") as file:
    file.write(f'Best_acc_Result: {best_acc}\n')
with open(f'{user_name}/{net_name}/{dataset}/{method}/scores_{net_name}.txt', "a+") as file:
    file.write(f'Best_edit_Result: {best_edit}\n')
with open(f'{user_name}/{net_name}/{dataset}/{method}/scores_{net_name}.txt', "a+") as file:
    file.write(f'Best_f1_Result: {best_f1}\n')