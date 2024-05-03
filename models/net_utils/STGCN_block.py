import torch.nn as nn
from models.net_utils.tgcn import ConvTemporalGraphical


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
        Args:
            in_channels (int): Number of channels in the input sequence data
            out_channels (int): Number of channels produced by the convolution
            kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
            stride (int, optional): Stride of the temporal convolution. Default: 1
            dropout (int, optional): Dropout rate of the final output. Default: 0
            residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
        Shape:
            - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
            - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` formatz
            - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
            - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
            where
                :math:`N` is a batch size,
                :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
                :math:`T_{in}/T_{out}` is a length of input/output sequence,
                :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 A=None,
                 dilation=1,
                 residual=True):
        super(st_gcn_block, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        pad = int((dilation*(kernel_size[0]-1))/2)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])


        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=(stride, 1),
                padding=(pad, 0),
                dilation=(dilation, 1),
            ),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)
        x = self.relu(x)
        x = x + res
        return x, A
