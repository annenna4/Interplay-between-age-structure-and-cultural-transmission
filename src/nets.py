import torch
import torch.nn as nn

import torch.nn.functional as F

from utils import get_arguments


class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        return conv1d_same_padding(
            input, self.weight, self.bias, self.stride, self.dilation, self.groups
        )


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = ((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups,
    )


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__()

        self.input_args = get_arguments()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, 128, 7, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        )
        self.final = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.final(x.mean(dim=-1))  # GAP


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        layer_dropout=0.0,
        feature_dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size,
            hidden_size // (1 + bidirectional),
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(num_layers > 1) * layer_dropout,
        )

        self.feature_dropout = feature_dropout

    def forward(self, x):
        x = torch.nn.functional.dropout(
            x, p=self.feature_dropout, training=self.training
        )
        out, _ = self.rnn(x)
        return out.view(-1)


class RNNFCN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        layer_dropout=0.0,
        feature_dropout=0.0,
        bidirectional=False,
    ):
        self.fcn = FCN(1)
        self.rnn = RNN(
            input_size,
            hidden_size,
            num_layers=num_layers,
            layer_dropout=layer_dropout,
            feature_dropout=feature_dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        return self.fcn(self.rnn(x))
