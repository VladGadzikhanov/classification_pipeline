from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.jit.annotations import List


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features)),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1", nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        ),
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module("relu2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ),
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNetAttn(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    # growth_rate=12, block_config=(16, 16, 16), num_init_features=24 => DenseNet 100
    # growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64 => DenseNet 121
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        use_attention=False,
        num_channels=3,
        num_classes=1000,
    ):

        super(DenseNetAttn, self).__init__()

        self.use_attention = use_attention
        self.num_channels = num_channels

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(num_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                    ),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        self.fc = nn.Linear(num_features, num_features // 2)

        if self.use_attention:
            self.attn = nn.Linear(num_features // 2, 1)

        # Linear layer
        self.classifier = nn.Linear(num_features // 2, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(0, 1)  # BGHW -> GBHW
        x = torch.stack([self.fc(torch.flatten(F.adaptive_avg_pool2d(F.relu(self.features(y)), (1, 1)), 1)) for y in x])
        x = x.transpose(0, 1)  # GBF -> BGF

        if self.use_attention:
            attn_weights = F.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(1)  # B1G
            x = attn_weights.bmm(x)  # BGF (X) B1G = B1F
            x = x.squeeze(1)  # MM of weights and features for each photo
        else:
            # BGF -> BF (sum over objs)
            x = x.sum(1)

        x = self.classifier(x)

        if self.use_attention:
            return x, attn_weights
        else:
            return x, None

    def one_shot_predict(self, batch, device):
        names, imgs_path, images, targets = batch
        with torch.no_grad():
            outputs, attn_weights = self.forward(images.to(device))
            attn_coefs = attn_weights.squeeze(1).to("cpu").numpy()
            batch_probs = F.softmax(outputs, dim=1).to("cpu")
            batch_preds = torch.max(batch_probs, 1)[1]
        tags = ["obj_id", "img_path", "true", "pred"]
        tags.extend([f"prob_{i}" for i in range(batch_probs.shape[1])])
        tags.extend([f"w_{i}" for i in range(attn_coefs.shape[1])])
        return (
            tags,
            np.c_[
                np.array(names),
                np.array(imgs_path),
                targets.numpy(),
                batch_preds.numpy(),
                batch_probs.numpy(),
                attn_coefs,
            ],
        )
