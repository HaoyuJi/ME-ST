import torch
import torch.nn as nn
import torch.nn.functional as F
from models.net_utils.tgcn import ConvTemporalGraphical
import einops
from torch.nn.init import constant_
from torch import optim

from  models.net_utils.pos_embed import Pos_Embed

import copy
import numpy as np
import math

class Encoder_bound(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha,joint_num,segment_num, A):
        super(Encoder_bound, self).__init__()
        self.conv_1x1 = nn.Conv2d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha, joint_num,segment_num, A) for i in # 2**i
             range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_bound = nn.Conv1d(num_f_maps, 1, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, T,V)
        :param mask:
        :return:
        '''

        # if self.channel_masking_rate > 0:
        #     x = self.dropout(x)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        N, C, T, V=feature.size()
        feature = F.avg_pool2d(feature, kernel_size=(1, V))

        # M pooling
        feature = feature.view(N, C, T) #（batch，channel，temporal）
        out = self.conv_out(feature) * mask[:, 0:1, :] #（batch，class，temporal）
        bound = self.conv_bound(feature) * mask[:, 0:1, :]

        return out, bound, feature

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha,joint_num,segment_num, A):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv2d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha, joint_num,segment_num, A) for i in # 2**i
             range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, T,V)
        :param mask:
        :return:
        '''

        # if self.channel_masking_rate > 0:
        #     x = self.dropout(x)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        N, C, T, V=feature.size()
        feature = F.avg_pool2d(feature, kernel_size=(1, V))

        # M pooling
        feature = feature.view(N, C, T) #（batch，channel，temporal）
        out = self.conv_out(feature) * mask[:, 0:1, :] #（batch，class，temporal）

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.Downlayers = nn.ModuleList((
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 0),
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 1),
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 2),
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 3),

        ))
        self.Midlayers = nn.ModuleList((
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 4),
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 4),
        ))
        self.Uplayers = nn.ModuleList((
            AttModule_Decoder(2 ** 3, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 3),
            AttModule_Decoder(2 ** 5, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 2),
            AttModule_Decoder(2 ** 7, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 1),
            AttModule_Decoder(2 ** 9, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 0),
        ))
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        mask_copy = mask
        masks = [mask_copy]

        for i in range(4):
            mask_copy = F.max_pool1d(mask_copy, kernel_size=2)
            masks.append(mask_copy)

        for index, layer in enumerate(self.Downlayers):
            feature = layer(feature, fencoder, masks[index])
            feature = F.avg_pool1d(feature, kernel_size=2)

        for index, layer in enumerate(self.Midlayers):
            feature = layer(feature, fencoder, masks[4])

        for index, layer in enumerate(self.Uplayers):
            feature = nn.functional.interpolate(feature, size=masks[3-index].shape[2], mode='linear')
            feature = layer(feature, fencoder, masks[3-index])


        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, joint_num,segment_num, A):
        super(AttModule, self).__init__()

        self.GCN_layer = ConvFeedForward(A, in_channels, out_channels, dilation)
        self.convForReshapeV = self.conv_1x1 = nn.Conv1d(out_channels * segment_num, out_channels, 1)

        self.Vatt_layer = Spatial_AttLayer(in_channels, out_channels, in_channels // r1,
                                         num_frames=1,
                                         num_joints= joint_num,
                                         num_heads=3,
                                         kernel_size= (1,1),
                                         use_pes=True,
                                         att_drop= 0)

        self.TCN_layer = nn.Sequential( nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(dilation, 0), dilation=(dilation, 1)),
                                    nn.GELU(), nn.BatchNorm2d(in_channels, track_running_stats=False)
                                    )

        self.convForReshapeT = self.conv_1x1 = nn.Conv1d(out_channels * joint_num, out_channels, 1)
        self.instance_norm = nn.BatchNorm1d(in_channels)
        self.instance_norm2d = nn.BatchNorm2d(in_channels)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type,
                                  stage=stage)  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.joint_num =joint_num
        self.alpha = alpha
        self.segment_num = segment_num
        self.A = A

        self.feed_forward1 = ConvFeedForward_withoutGCN(joint_num, in_channels * 2, out_channels)
        self.feed_forward2 = ConvFeedForward_withoutGCN(joint_num, in_channels*2, out_channels)
        self.instance_norm2d2 = nn.BatchNorm2d(64)

    def forward(self, x, f, mask):
        gcn_out, _ = self.GCN_layer(x, self.A)

        N, C, T, V = gcn_out.size()  # N：batch C: channel T:temporal V:joint
        out = F.avg_pool2d(gcn_out, kernel_size=(T // self.segment_num, 1))
        out = out.permute(0, 2, 1, 3).contiguous().view(N * self.segment_num, C, V)  # （N*64，C，V）
        out = out.unsqueeze(2)  # (N, C * 64, 1, V)

        out = self.Vatt_layer(out)
        out = out.squeeze(2)
        out = out.view(N, self.segment_num, C, V).permute(0, 2, 1, 3).contiguous()  # （N*64，C，V）

        out = nn.functional.interpolate(out, size=(T, V), mode='bilinear')
        out = torch.concat((gcn_out, out), dim=1)
        out = self.feed_forward1(out)


        tcn_out = self.TCN_layer(out)
        out = tcn_out.permute(0, 1, 3, 2).contiguous()  # （N,C,V,T）
        out = out.view(N, V * C, T)


        out = self.convForReshapeT(out)

        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        out = out.unsqueeze(-1).expand(-1,-1,-1,self.joint_num)
        out = self.instance_norm2d(out)
        out = torch.concat((tcn_out, out),dim=1)
        out = self.feed_forward2(out)
        out = x + out
        out = self.instance_norm2d2(out)
        return  out * (mask[:, 0:1, :].unsqueeze(-1))


class AttModule_Decoder(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, downrate):
        super(AttModule_Decoder, self).__init__()
        self.feed_forward = ConvFeedForward_Decoder(dilation, in_channels, out_channels)
        self.instance_norm = nn.BatchNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type,
                                  stage=stage, downrate= downrate)  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]



class ConvFeedForward(nn.Module):
    def __init__(self, A, in_channels, out_channels,dilation):
        super(ConvFeedForward, self).__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, 3)

        self.edge_importance = nn.Parameter(torch.ones(A.size()))



    def _reset_parameters(self):
        constant_(self.layer[0].bias, 0.)

    def forward(self, x, A):
        out, A = self.gcn(x,A * self.edge_importance) #gcn


        return out, A


class ConvFeedForward_withoutGCN(nn.Module):
    def __init__(self, joint_num, in_channels, out_channels):
        super(ConvFeedForward_withoutGCN, self).__init__()

        self.convRS = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), dilation=(1, 1)),
            nn.GELU(), nn.BatchNorm2d(out_channels, track_running_stats=False)

            )

    def _reset_parameters(self):
        constant_(self.layer[0].bias, 0.)

    def forward(self, x):
        out = self.convRS(x)  # gcn
        return out

class ConvFeedForward_Decoder(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward_Decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.GELU()
        )

    def forward(self, x):
        return self.layer(x)


class Spatial_AttLayer(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size, use_pes=True, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)
        # padt = int((kernel_size[0] - 1) / 2)

        # Spatio-Temporal Tuples Attention
        if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))


        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):

        N, C, T, V = x.size()
        xs = self.pes(x) + x if self.use_pes else x
        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        attention = self.tan(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas
        attention = attention + self.att0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T, V)
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)

        return xs

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, dilaion, stage, att_type, downrate=1):  # r1 = r2
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)

        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.downrate = downrate
        self.dilaion = dilaion
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder', 'decoder']

        self.softmax = nn.Softmax(dim=-1)



    def forward(self, x1, x2, mask):
        # x1 from the current stage
        # x2 from the last stage

        query = self.query_conv(x1)

        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
            key = self.key_conv(x2)
        else:
            value = self.value_conv(x1)
            key = self.key_conv(x1)

        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            if self.stage == 'encoder':
                return self._sliding_window_self_att(query, key, value, mask)
            elif self.stage == 'decoder':
                return self._sliding_window_self_att_cross(query, key, value, mask)


    def _sliding_window_self_att(self, q, k, v, mask):
        QB, QE, QS = q.size()
        KB, KE, KS = k.size()
        VB, VE, VS = v.size()

        # padding zeros for the last segment
        # we want our sequence be dividable by  self.dilaion, so we need QS % self.dilaion == 0, if it is not the case we will pad it so it become
        nb = QS // self.dilaion
        if QS % self.dilaion != 0:
            q = F.pad(q, pad=(0, self.dilaion - QS % self.dilaion), mode='constant', value=0)
            k = F.pad(k, pad=(0, self.dilaion - QS % self.dilaion), mode='constant', value=0)
            v = F.pad(v, pad=(0, self.dilaion - QS % self.dilaion), mode='constant', value=0)
            nb += 1

        padding_mask = torch.cat([torch.ones((QB, 1, QS)).to(q.device) * mask[:, 0:1, :],
                                  torch.zeros((QB, 1, self.dilaion * nb - QS)).to(q.device)],
                                 dim=-1)

        # sliding window approach, by splitting query_proj and key_proj into shape (QE, l) x (QE, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(QB, QE, nb, self.dilaion).permute(0, 2, 1, 3).reshape(QB, nb, QE,
                                                                            self.dilaion)

        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = F.pad(k, pad=(self.dilaion // 2, self.dilaion // 2), mode='constant',
                  value=0)
        v = F.pad(v, pad=(self.dilaion // 2, self.dilaion // 2), mode='constant', value=0)
        padding_mask = F.pad(padding_mask, pad=(self.dilaion // 2, self.dilaion // 2), mode='constant',
                             value=0)


        # 2. reshape key_proj of shape (QB*nb, QE, 2*self.dilaion)
        k = torch.cat([k[:, :, i * self.dilaion:(i + 1) * self.dilaion + (self.dilaion // 2) * 2].unsqueeze(1) for i in
                       range(nb)],
                      dim=1)
        v = torch.cat([v[:, :, i * self.dilaion:(i + 1) * self.dilaion + (self.dilaion // 2) * 2].unsqueeze(1) for i in
                       range(nb)], dim=1)

        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat(
            [padding_mask[:, :, i * self.dilaion:(i + 1) * self.dilaion + (self.dilaion // 2) * 2].unsqueeze(1) for i in
             range(nb)], dim=1)  # （batch，1，temporal）变为（batch,temporal/dilaion，1，2×dilaion）

        # construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        window_mask = torch.zeros((1, self.dilaion, self.dilaion + 2 * (self.dilaion // 2))).to(
            q.device)  # window_mask （1,dilaion,dilaion×2）
        for i in range(self.dilaion):
            window_mask[:, :, i:i + self.dilaion] = 1

        final_mask = window_mask.unsqueeze(1).repeat(QB, nb, 1, 1) * padding_mask

        proj_query = q  # （batch，T/Dilaion，channel，Dilaion）
        proj_key = k  # （batch，T/Dilaion，channel，2×Dilaion）
        proj_val = v  # （batch，T/Dilaion，channel，2×Dilaion）
        padding_mask = final_mask

        b, m, QE, l1 = proj_query.shape
        b, m, KE, l2 = proj_key.shape

        energy = torch.einsum('n b k i, n b k j -> n b i j', proj_query,
                              proj_key)
        attention = energy / (np.sqrt(QE) * 1.0)
        attention = attention + torch.log( padding_mask + 1e-6)
        attention = self.softmax(attention)
        attention = attention * padding_mask
        output = torch.einsum('n b i k, n b j k-> n b i j', proj_val,
                              attention)

        bb, cc, ww, hh = output.shape
        output = einops.rearrange(output, 'b c h w -> (b c) h w')
        output = self.conv_out(F.gelu(output))
        output = einops.rearrange(output, '(b c) h w->b c h w', b=bb, c=cc)

        output = output.reshape(QB, nb, -1, self.dilaion).permute(0, 2, 1, 3).reshape(QB, -1,
                                                                                      nb * self.dilaion)
        output = output[:, :, 0:QS]
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att_cross(self, q, k, v, mask):
        QB, QE, QS = q.size()
        KB, KE, KS = k.size()
        VB, VE, VS = v.size()

        Q_dilaion = self.dilaion
        KV_dilaion = self.dilaion * self.downrate

        nb = QS // Q_dilaion
        if QS % Q_dilaion != 0:
            q = F.pad(q, pad=(0, Q_dilaion - QS % Q_dilaion), mode='constant', value=0)
            nb += 1
        if KS % KV_dilaion != 0:
            k = F.pad(k, pad=(0, KV_dilaion - KS % KV_dilaion), mode='constant', value=0)
        if VS % KV_dilaion != 0:
            v = F.pad(v, pad=(0, KV_dilaion - VS % KV_dilaion), mode='constant', value=0)


        padding_mask = torch.cat([torch.ones((QB, 1, QS)).to(q.device) * mask[:, 0:1, :],
                                  torch.zeros((QB, 1, Q_dilaion * nb - QS)).to(q.device)],
                                 dim=-1)

        q = q.reshape(QB, QE, nb, Q_dilaion).permute(0, 2, 1, 3).reshape(QB, nb, QE,
                                                                            Q_dilaion)

        k = F.pad(k, pad=(KV_dilaion // 2, KV_dilaion // 2), mode='constant',
                  value=0)
        v = F.pad(v, pad=(KV_dilaion // 2, KV_dilaion // 2), mode='constant', value=0)
        padding_mask = F.pad(padding_mask, pad=(Q_dilaion // 2, Q_dilaion // 2), mode='constant',
                             value=0)


        k = torch.cat([k[:, :, i * KV_dilaion:(i + 1) * KV_dilaion + (KV_dilaion // 2) * 2].unsqueeze(1) for i in
                       range(nb)],
                      dim=1)
        v = torch.cat([v[:, :, i * KV_dilaion:(i + 1) * KV_dilaion + (KV_dilaion // 2) * 2].unsqueeze(1) for i in
                       range(nb)], dim=1)

        # 3. construct window mask
        padding_mask = torch.cat(
            [padding_mask[:, :, i * Q_dilaion:(i + 1) * Q_dilaion].unsqueeze(1) for i in
             range(nb)], dim=1).permute(0, 1, 3, 2).contiguous()

        # construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        window_mask = torch.zeros((1, Q_dilaion, KV_dilaion + 2 * (KV_dilaion // 2))).to(
            q.device)  # window_mask （1,dilaion,dilaion×2）
        for i in range(self.dilaion):
            window_mask[:, :, i:i + KV_dilaion] = 1

        final_mask = window_mask.unsqueeze(1).repeat(QB, nb, 1,
                                                     1) * padding_mask

        proj_query = q
        proj_key = k
        proj_val = v
        padding_mask = final_mask

        b, m, QE, l1 = proj_query.shape
        b, m, KE, l2 = proj_key.shape

        energy = torch.einsum('n b k i, n b k j -> n b i j', proj_query,
                              proj_key)
        attention = energy / (np.sqrt(QE) * 1.0)
        attention = attention + torch.log(
            padding_mask + 1e-6)
        attention = self.softmax(attention)
        attention = attention * padding_mask
        output = torch.einsum('n b i k, n b j k-> n b i j', proj_val,
                              attention)

        bb, cc, ww, hh = output.shape
        output = einops.rearrange(output, 'b c h w -> (b c) h w')
        output = self.conv_out(F.gelu(output))
        output = einops.rearrange(output, '(b c) h w->b c h w', b=bb, c=cc)

        output = output.reshape(QB, nb, -1, self.dilaion).permute(0, 2, 1, 3).reshape(QB, -1,
                                                                                      nb * self.dilaion)
        output = output[:, :, 0:QS]
        return output * mask[:, 0:1, :]