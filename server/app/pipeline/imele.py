"""IMELE building-height backbone: SENet154 encoder + D2/MFF/R decoder.

Vendored from https://github.com/speed8928/IMELE (checkpoint
Block0_skip_model_110.pth.tar). The upstream repo cannot run as released:
`senet.py` imports the nonexistent `harmonic` package and `models/net.py` has a
debug `x_block0.view(-1,250,250)` that crashes for any input other than 500px.
This module reproduces the same architecture/forward-pass (from the released
`models/modules.py`, byte-for-byte in behaviour) with imports cleaned and the
debug noise removed. The checkpoint state_dict carries `E.Harm.*` keys from an
older model variant that the released encoder never defines or uses; those keys
are dropped on load, and any other unexpected/missing key is an error.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 440
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---- SENet154 encoder (senet.py, without pretrained weights / harmonic) ----


class _SENet(nn.Module):
    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        super().__init__()
        self.inplanes = inplanes

        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential()
        for name, mod in layer0_modules:
            setattr(self.layer0, name, mod)
            self.layer0._modules[name] = mod

        self.layer1 = self._make_layer(
            block, planes=64, blocks=layers[0], groups=groups, reduction=reduction,
            downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(
            block, planes=128, blocks=layers[1], stride=2, groups=groups,
            reduction=reduction, downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(
            block, planes=256, blocks=layers[2], stride=2, groups=groups,
            reduction=reduction, downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(
            block, planes=512, blocks=layers[3], stride=2, groups=groups,
            reduction=reduction, downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)


class _SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class _SEBottleneck(nn.Module):
    """Bottleneck for SENet154."""
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = _SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


# ---- decoder blocks (models/modules.py) ----


class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)
        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out


class _E_senet(nn.Module):
    def __init__(self, original_model, num_features=2048):
        super().__init__()
        self.base = nn.Sequential(*list(original_model.children())[:-3])
        self.pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.down = _UpProjection(64, 128)

    def forward(self, x):
        # block0 skip path: first 6 of layer0 at 1/2 res, rest of layer0 -> 1/4.
        # The released upstream code comments out the Harm/pool/down lines; the
        # checkpoint's E.Harm.* keys are legacy and dropped on load.
        x_block0 = self.base[0][0:6](x)
        x = self.base[0][6:](x_block0)
        x_block1 = self.base[1](x)
        x_block2 = self.base[2](x_block1)
        x_block3 = self.base[3](x_block2)
        x_block4 = self.base[4](x_block3)
        return x_block0, x_block1, x_block2, x_block3, x_block4


class _D2(nn.Module):
    def __init__(self, num_features=2048):
        super().__init__()
        self.conv = nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)
        self.up1 = _UpProjection(1024, 512)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.up2 = _UpProjection(512, 256)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.up3 = _UpProjection(256, 128)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.up4 = _UpProjection(128, 64)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x_block0, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_block3 = F.relu(self.bn1(self.conv1(x_block3)))
        cx_d1 = torch.cat((x_d1, x_block3), 1)
        cx_d1 = F.relu(self.bn1(self.conv1(cx_d1)))

        x_d2 = self.up2(cx_d1, [x_block2.size(2), x_block2.size(3)])
        x_block2 = F.relu(self.bn2(self.conv2(x_block2)))
        cx_d2 = torch.cat((x_d2, x_block2), 1)
        cx_d2 = F.relu(self.bn2(self.conv2(cx_d1)))  # released quirk, keep byte-for-byte

        x_d3 = self.up3(cx_d2, [x_block1.size(2), x_block1.size(3)])
        x_block1 = F.relu(self.bn3(self.conv3(x_block1)))
        cx_d3 = torch.cat((x_d3, x_block1), 1)
        cx_d3 = F.relu(self.bn3(self.conv3(cx_d3)))

        x_d4 = self.up4(cx_d3, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        cx_d4 = torch.cat((x_d4, x_block0), 1)
        cx_d4 = F.relu(self.bn3(self.conv4(cx_d4)))
        return cx_d4


class _MFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super().__init__()
        self.up0 = _UpProjection(num_input_features=64, num_output_features=16)
        self.up1 = _UpProjection(num_input_features=block_channel[0], num_output_features=16)
        self.up2 = _UpProjection(num_input_features=block_channel[1], num_output_features=16)
        self.up3 = _UpProjection(num_input_features=block_channel[2], num_output_features=16)
        self.up4 = _UpProjection(num_input_features=block_channel[3], num_output_features=16)
        self.conv = nn.Conv2d(80, 80, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(80)

    def forward(self, x_block0, x_block1, x_block2, x_block3, x_block4, size):
        x_m0 = self.up0(x_block0, size)
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)
        x = self.bn(self.conv(torch.cat((x_m0, x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)
        return x


class _R(nn.Module):
    def __init__(self, block_channel):
        super().__init__()
        self.conv0 = nn.Conv2d(208, 144, kernel_size=1, stride=1)
        self.bn0 = nn.BatchNorm2d(144)
        self.conv1 = nn.Conv2d(144, 144, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(144)
        self.conv2 = nn.Conv2d(144, 144, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv3 = nn.Conv2d(144, 72, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(72)
        self.conv4 = nn.Conv2d(72, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        return self.conv4(x)


class _ImeleModel(nn.Module):
    """net.model from upstream models/net.py, debug lines removed."""

    def __init__(self, num_features=2048, block_channel=(256, 512, 1024, 2048)):
        super().__init__()
        self.E = _E_senet(_senet154())
        self.D2 = _D2(num_features=num_features)
        self.MFF = _MFF(list(block_channel))
        self.R = _R(list(block_channel))

    def forward(self, x):
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D2(x_block0, x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block0, x_block1, x_block2, x_block3, x_block4,
                         [x_decoder.size(2), x_decoder.size(3)])
        return self.R(torch.cat((x_decoder, x_mff), 1))


def _senet154():
    return _SENet(_SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2)


def load_model(checkpoint_path: str) -> _ImeleModel:
    """Build the released SENet154 architecture and load the IMELE checkpoint.

    The pretrained checkpoint is a .tar whose ``state_dict`` is saved without a
    DataParallel ``module.`` prefix. Its ``E.Harm.*`` keys belong to an older
    model variant the released encoder never defines; they are dropped. Any
    other unexpected or missing key is a real architecture mismatch and raises.
    """
    model = _ImeleModel()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("E.Harm.")]
    if unexpected or missing:
        raise ValueError(
            f"IMELE checkpoint/architecture mismatch: unexpected={unexpected}, "
            f"missing={missing}")
    model.eval()
    return model