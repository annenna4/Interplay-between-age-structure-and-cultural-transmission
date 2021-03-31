from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import tsai.all as tsai


class ResNet(tsai.ResNet):
    def __init__(self, c_in):
        super().__init__(c_in, 1)
        self.fc = nn.Identity()
        self.output_size = 64 * 2  # hard coded in tsai...


class FCN(tsai.FCN):
    def __init__(self, c_in, layers=(128, 256, 128), kss=(7, 5, 3)):
        super().__init__(c_in, 1, layers, kss)
        self.fc = nn.Identity()
        self.output_size = layers[-1]


class InceptionTime(tsai.InceptionTime):
    def __init__(self, c_in, nf=32, nb_filters=None, **kwargs):
        super().__init__(c_in, 1, nf, nb_filters, **kwargs)
        self.fc = nn.Identity()
        self.output_size = nf * 4  # hard-coded in tsai


def make_mlp_layer(dropout, input_size, output_size):
    mappings = [nn.ReLU(inplace=True)]
    if dropout > 0:
        mappings.append(nn.Dropout(dropout))
    mappings.append(nn.Linear(input_size, output_size))
    return nn.Sequential(*mappings)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, layers=(128, 128, 128), dropout=0.0):
        super().__init__()

        mappings = [nn.Linear(input_size, layers[0])]
        for i in range(1, len(layers)):
            mappings.append(make_mlp_layer(dropout, layers[i - 1], layers[i]))

        mappings.append(nn.ReLU(inplace=True))
        mappings.append(nn.Linear(layers[-1], 1))
        self.mappings = nn.Sequential(*mappings)

    def forward(self, x):
        return self.mappings(x)


class TimeseriesClassifier(nn.Module):
    def __init__(self, classifier, embedder=None):
        super().__init__()

        self.embedder = embedder
        self.classifier = classifier

    def forward(self, x):
        if self.embedder is not None:
            x = self.embedder(x)
        logits = self.classifier(x)
        likelihood = torch.sigmoid(logits)
        return likelihood.view(-1)
