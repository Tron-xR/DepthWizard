# flake8: noqa
# (vendored file - upstream formatting is preserved so the published state
# dicts stay loadable; only the header note and _load_dinov2() are ours)
# Vendored from SynRS3D (https://github.com/JTRNEO/SynRS3D) at commit
# ab5a4857825448e4c5cd1fe3aa3045c7348814c5, files models/dpt.py and
# models/blocks.py merged into one module. MIT Licence, (c) Jian Song
# ("SynRS3D: A Synthetic Dataset for Global-scale Remote Sensing 3D
# Perception", NeurIPS 2024 D&B, arXiv:2406.18151).
#
# Local modifications (kept minimal so upstream checkpoints load unchanged):
#   1. the two upstream modules are merged here (import path removed);
#   2. unused imports (argparse, huggingface_hub.PyTorchModelHubMixin) dropped;
#   3. the DINOv2 torch.hub load is routed through _load_dinov2(): the cached
#      worktree copy is the first candidate, the zip is extracted in place,
#      and only if neither exists does it download; no silent re-fetch of the
#      mutable 'main' ref once a copy is on disk;
#   4. dinov2_source_info() reports the commit sha + zip sha256 of that copy so
#      run provenance pins the backbone code as well as the checkpoint bytes.
# Do not "improve" the layer structure: it must stay byte-compatible with the
# published state dicts (RS3DAda_vitl_DPT_height.pth, altimap best.pth).
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict

# Worktree-local torch.hub cache (server/app/pipeline -> repo root). The smoke
# script and runs keep TORCH_HOME here; this path is preferred over the user
# cache so the pinned DINOv2 copy travels with the worktree.
_WORKTREE_HUB = Path(__file__).resolve().parents[3] / ".torch-cache" / "hub"
_DINOV2_DIR = "facebookresearch_dinov2_main"


def _cached_dinov2_dirs():
    yield _WORKTREE_HUB / _DINOV2_DIR
    import torch.hub

    yield Path(torch.hub.get_dir()) / _DINOV2_DIR


def _load_dinov2(encoder: str):
    """dinov2_{encoder}14 with pretrained=False.

    The checkpoint carries the full encoder weights; this call only provides
    the module code. Load order: extracted worktree copy -> extract the
    worktree ``main.zip`` in place -> extracted TORCH_HOME copy -> download.
    """
    import torch.hub
    import zipfile

    name = "dinov2_{:}14".format(encoder)
    for d in _cached_dinov2_dirs():
        if d.is_dir():
            return torch.hub.load(str(d), name, source="local", pretrained=False)
    zip_path = _WORKTREE_HUB / "main.zip"
    if zip_path.is_file():
        target = _WORKTREE_HUB / _DINOV2_DIR
        with zipfile.ZipFile(zip_path) as zf:
            top = zf.namelist()[0].split("/", 1)[0]  # 'dinov2-<commit sha>'
            zf.extractall(_WORKTREE_HUB)
        (_WORKTREE_HUB / top).rename(target)
        return torch.hub.load(str(target), name, source="local", pretrained=False)
    return torch.hub.load("facebookresearch/dinov2", name, pretrained=False,
                          trust_repo=True)


def dinov2_source_info() -> dict:
    """Pin for the cached DINOv2 code: commit sha (from the zipball top dir)
    + sha256 of the zip itself. Both None if no worktree copy exists."""
    import hashlib
    import zipfile

    info = {"utils_zip": str(_WORKTREE_HUB / "main.zip"),
            "zip_sha256": None, "revision": None,
            "cache_dir": str(_WORKTREE_HUB / _DINOV2_DIR)}
    zip_path = _WORKTREE_HUB / "main.zip"
    if zip_path.is_file():
        h = hashlib.sha256()
        with open(zip_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        info["zip_sha256"] = h.hexdigest()
        try:
            with zipfile.ZipFile(zip_path) as zf:
                top = zf.namelist()[0].split("/", 1)[0]
            info["revision"] = "facebookresearch/dinov2@" + top.rsplit("-", 1)[-1]
        except (zipfile.BadZipFile, IndexError):
            pass
    return info


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        if self.nclass > 1:
            out = self.scratch.output_conv(path_1)
        else:
            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
            out = self.scratch.output_conv2(out)
        
        return out
        
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', head_configs: Optional[List[Dict]] = None, features=256, out_channels=[256, 512, 1024, 1024], use_clstoken=False, pretrained=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        
        self.pretrained = _load_dinov2(encoder)
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.head_configs = head_configs or []
        self.heads = nn.ModuleDict()
        
        # Dynamically creating heads based on configurations
        for config in self.head_configs:
            task = config['name']
            use_bn = True
            nclass = config.get('nclass', None)
            head = DPTHead(nclass, dim, features, use_bn, out_channels, use_clstoken)
            self.heads[task] = head
        
    def forward(self, x):
        h, w = x.shape[-2:]
        
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14
        outputs = {}

        for task, head in self.heads.items():
            output = head(features, patch_h, patch_w)

            output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=True)
                
            outputs[task] = output
            
        return outputs
