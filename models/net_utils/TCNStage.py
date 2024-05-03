import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers_per_stage, input_dim, num_f_maps, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage_0 = SingleStageModel(num_layers=num_layers_per_stage, input_dim=input_dim, num_f_maps = num_f_maps, num_classes=num_classes, dilations=[2**i for i in range(num_layers_per_stage)])
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(
            num_layers=num_layers_per_stage, input_dim=input_dim, num_f_maps = num_f_maps,
            num_classes=num_classes, dilations=[2**i for i in range(num_layers_per_stage)]))
            for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage_0(x, mask) * mask[:, 0:1, :]
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self,  num_layers, input_dim, num_f_maps, num_classes, dilations):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(dilations[i], num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out

class SingleStageTCN(nn.Module):
    def __init__(self,  num_layers, input_dim, num_f_maps, num_classes, dilations):
        super(SingleStageTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer_ASRF(dilations[i], num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out



class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation) #膨胀卷积
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, mask):
        out = self.conv_dilated(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out + x
        return out * mask[:, 0:1, :]

class DilatedResidualLayer_ASRF(nn.Module):
    def __init__(self, dilation: int, in_channel: int, out_channels: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        # out = self.bn(out)
        out = self.dropout(out)

        return (x + out) * mask[:, 0:1, :]