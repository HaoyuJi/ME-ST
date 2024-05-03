import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.net_utils.STGCN_block import st_gcn_block
from models.net_utils.tgcn import ConvTemporalGraphical
from models.net_utils.graph import Graph
from models.net_utils.TCNStage import MultiStageModel
from models.net_utils.PyramidTransformer import Encoder
from models.net_utils.PyramidTransformer import Decoder
import math

class Model(nn.Module):
    r""".
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, args):
        super(Model, self).__init__()
        in_channels = args.channel
        feature_dim = args.feat_dim
        dil = args.temporal_dil
        filters = args.spatial_filters
        edge_importance_weighting = args.edge_importance_weighting #True
        n_classes = args.num_classes
        graph_args = {'layout': args.graph_layout, 'strategy': args.graph_strategy}
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).to('cuda')
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # build networks
        self.conv_1x1 = nn.Conv2d(in_channels, filters, 1)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.tcns = MultiStageModel(num_stages=args.num_stages,num_layers_per_stage=args.num_layers,input_dim=args.feat_dim,num_f_maps=args.num_f_maps, num_classes=args.num_classes)

        self.encoder = Encoder(num_layers=args.num_layers, r1=2, r2=2, num_f_maps=args.num_f_maps,
                               input_dim=args.num_f_maps, num_classes=args.num_classes, channel_masking_rate=0.3,
                               att_type='sliding_att', alpha=1,joint_num=args.joint_num, segment_num=args.segment_num, A=self.A)
        self.decoders = nn.ModuleList(
            [copy.deepcopy(Decoder(num_layers=args.num_layers, r1=2, r2=2, num_f_maps=args.num_f_maps,
                                   input_dim=args.num_classes, num_classes=args.num_classes, att_type='sliding_att',
                    alpha=math.exp(-1 * s))) for s in range(args.num_stages)])  # num_decoders


    def forward(self, x, mask):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)


        # forward
        x = self.conv_1x1(x)
        out, feature = self.encoder(x, mask)

        outputs = [out]

        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask)
            outputs.append(out) # 5list（batch，class，temporal）

        return outputs


