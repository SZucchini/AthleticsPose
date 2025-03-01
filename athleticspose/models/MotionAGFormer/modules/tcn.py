"""Temporal Convolutional Network module from TaatiTeam/MotionAGFormer repository (Apache-2.0 License)."""

import torch
import torch.nn as nn


class TemporalConv(nn.Module):
    """Temporal Convolutional Network."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        """Initialize TemporalConv."""
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Forward pass of TemporalConv."""
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScaleTCN(nn.Module):
    """Multi-Scale Temporal Convolutional Network."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        dilations=(1, 2),
        residual=True,
        residual_kernel_size=1,
    ):
        """Initialize MultiScaleTCN."""
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, "# out channels should be multiples of # branches (6x)"

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        # Temporal Convolution branches
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=ks,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
                for ks, dilation in zip(kernel_size, dilations, strict=False)
            ]
        )

        # Additional Max & 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),
            )
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(branch_channels),
            )
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(
                in_channels,
                out_channels,
                kernel_size=residual_kernel_size,
                stride=stride,
            )

    def forward(self, x):
        """Forward pass of MultiScaleTCN."""
        x = x.permute(0, 3, 1, 2)  # (B, T, J, C) -> (B, C, T, J)

        res = self.residual(x)
        branch_outs = []
        for temp_conv in self.branches:
            out = temp_conv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res

        out = out.permute(0, 2, 3, 1)  # (B, C, T, J) -> (B, T, J, C)
        return out
