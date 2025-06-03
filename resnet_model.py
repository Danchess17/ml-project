from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from odk import ODKConv2d


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    odk: bool = True,
    r: int = 1,
    num_matrices: int = 1,
) -> Union[nn.Conv2d, ODKConv2d]:
    if odk:
        return ODKConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            r=r,
            num_matrices=num_matrices,
        )
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        odk: bool = True,
        r: int = 1,
        num_matrices: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(
            in_channels,
            out_channels,
            stride,
            odk=odk,
            r=r,
            num_matrices=num_matrices,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(
            out_channels,
            out_channels,
            odk=odk,
            r=r,
            num_matrices=num_matrices,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        num_classes: int = 1000,
        odk_layers: List[int] = [3, 4],
        r: int = 1,
        num_matrices: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=3, stride=2, padding=3, bias=False
        )  # why pad=3
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            64,
            blocks=layers[0],
            odk=(1 in odk_layers),
            r=r,
            num_matrices=num_matrices,
        )
        self.layer2 = self._make_layer(
            128,
            blocks=layers[1],
            stride=2,
            odk=(2 in odk_layers),
            r=r,
            num_matrices=num_matrices,
        )
        self.layer3 = self._make_layer(
            256,
            blocks=layers[2],
            stride=2,
            odk=(3 in odk_layers),
            r=r,
            num_matrices=num_matrices,
        )
        self.layer4 = self._make_layer(
            512,
            blocks=layers[3],
            stride=2,
            odk=(4 in odk_layers),
            r=r,
            num_matrices=num_matrices,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int = 1,
        odk: bool = True,
        r: int = 1,
        num_matrices: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                odk=odk,
                r=r,
                num_matrices=num_matrices,
            )
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.in_channels,
                    out_channels,
                    odk=odk,
                    r=r,
                    num_matrices=num_matrices,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(
    layers: List[int],
    **kwargs: Any,
) -> Resnet:
    return Resnet(layers, **kwargs)


def resnet18(**kwargs: Any) -> Resnet:
    return _resnet([2, 2, 2, 2], **kwargs)
