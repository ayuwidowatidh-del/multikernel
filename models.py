"""NPCNN and MK-NPCNN.

The padded NPCNN and MK-NPCNN use the same strides, dropout, padding/crop and
classifier head. MK-NPCNN has three kernel sizes per stage instead of one.
"""
import torch
import torch.nn as nn
import config as C

def valid_len(n, k, s):
    """Output length of an unpadded conv, used as the crop length."""
    return (n - k) // s + 1


def baseline_lengths(input_len=C.INPUT_LEN, stages=C.BASE_STAGES):
    """Feature length after each stage of the original (unpadded) NPCNN."""
    lengths, n = [], input_len
    for k, s in stages:
        n = valid_len(n, k, s)
        lengths.append(n)
    return lengths


def centre_crop(x, target):
    """Crop a (B, C, L) tensor to `target` length, centred."""
    L = x.shape[-1]
    if L == target:
        return x
    if L < target:
        raise ValueError(f"cannot crop {L} up to {target}")
    start = (L - target) // 2
    return x[..., start:start + target]

class SingleKernelBlock(nn.Module):
    """Baseline stage: one kernel, symmetric padding, centre-cropped."""

    def __init__(self, in_ch, out_ch, kernel, stride, target_len, dropout):
        super().__init__()
        # even kernels as in the original NPCNN; the crop handles the extra point
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride, padding=kernel // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.target_len = target_len

    def forward(self, x):
        return self.drop(self.relu(centre_crop(self.conv(x), self.target_len)))


class MultiKernelBlock(nn.Module):
    """Three parallel convolutions (narrow / medium / broad kernel), concatenated.
    branch_mask and channel_mask are only used for the ablation at test time."""

    def __init__(self, in_ch, kernels, branch_ch, stride, target_len, dropout):
        super().__init__()
        assert len(kernels) == len(branch_ch) == 3
        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, ch, k, stride, padding=k // 2)
            for k, ch in zip(kernels, branch_ch)
        ])
        self.branch_ch = branch_ch
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.target_len = target_len
        self.register_buffer("branch_mask", torch.ones(3))
        self.channel_mask = None

    def forward(self, x):
        outs = []
        for i, b in enumerate(self.branches):
            o = b(x)
            if float(self.branch_mask[i]) != 1.0:
                o = o * self.branch_mask[i]
            outs.append(o)
        L = min(o.shape[-1] for o in outs)     # same length for odd kernels
        out = torch.cat([o[..., :L] for o in outs], dim=1)
        if self.channel_mask is not None:
            out = out * self.channel_mask.view(1, -1, 1).to(out.device)
        return self.drop(self.relu(centre_crop(out, self.target_len)))

class ValidKernelBlock(nn.Module):
    """Original Salgado stage: unpadded convolution, no crop."""

    def __init__(self, in_ch, out_ch, kernel, stride, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.relu(self.conv(x)))


class NPCNN(nn.Module):
    """padded=True: padding + centre crop (baseline for MK-NPCNN).
    padded=False: original NPCNN with valid convolutions.
    Both give feature lengths 1681 / 327 / 152."""

    def __init__(self, in_channels=1, padded=True):
        super().__init__()
        lens = baseline_lengths()
        ch_in, blocks = in_channels, []
        for (k, s), L in zip(C.BASE_STAGES, lens):
            blocks.append(SingleKernelBlock(ch_in, C.CHANNELS, k, s, L, C.DROPOUT_CONV)
                          if padded else
                          ValidKernelBlock(ch_in, C.CHANNELS, k, s, C.DROPOUT_CONV))
            ch_in = C.CHANNELS
        self.stages = nn.Sequential(*blocks)
        self.out_len = lens[-1]

    def forward(self, x):
        return self.stages(x)


class MKNPCNN(nn.Module):
    def __init__(self, in_channels=1, stages=None):
        super().__init__()
        stages = stages or C.MK_STAGES
        lens = baseline_lengths()
        ch_in, blocks = in_channels, []
        for (kernels, s), L in zip(stages, lens):
            blocks.append(MultiKernelBlock(ch_in, kernels, C.MK_BRANCH_CH, s, L,
                                           C.DROPOUT_CONV))
            ch_in = sum(C.MK_BRANCH_CH)
        self.stages = nn.Sequential(*blocks)
        self.out_len = lens[-1]

    def forward(self, x):
        return self.stages(x)
    
    def set_branch_mask(self, mask):
        """mask: iterable of 3 floats, e.g. (0, 1, 1) to disable the narrow branch."""
        for blk in self.stages:
            blk.branch_mask = torch.tensor([float(m) for m in mask],
                                           device=blk.branch_mask.device)

    def set_channel_mask(self, mask):
        """mask: (80,) tensor or None. Control condition for branch ablation."""
        for blk in self.stages:
            blk.channel_mask = mask

    def reset_masks(self):
        self.set_branch_mask((1, 1, 1))
        self.set_channel_mask(None)

class Predictor(nn.Module):
    """Classifier head, identical for both models (Salgado / Park dimensions)."""

    def __init__(self, in_features, out_features, dropout=C.DROPOUT_MLP):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 2300), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2300, 1150), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1150, out_features),
        )

    def forward(self, x):
        return self.MLP(x)


class Model(nn.Module):
    def __init__(self, arch="npcnn", num_classes=C.NUM_CLASSES, input_len=C.INPUT_LEN):
        super().__init__()
        self.arch, self.input_len = arch, input_len
        if arch == "npcnn":
            self.cnn = NPCNN(padded=True)       # padding-matched baseline
        elif arch == "npcnn_valid":
            self.cnn = NPCNN(padded=False)      # published Salgado NPCNN
        elif arch == "mknpcnn":
            self.cnn = MKNPCNN()
        else:
            raise ValueError(f"unknown arch '{arch}'")
        with torch.no_grad():
            feat = self.cnn(torch.zeros(1, 1, input_len))
        self.feat_shape = tuple(feat.shape[1:])
        self.MLP = Predictor(int(feat.numel()), num_classes)

    def forward(self, x):
        if x.shape[-1] != self.input_len:
            raise ValueError(
                f"expected input length {self.input_len}, got {x.shape[-1]}"
            )
        return self.MLP(self.cnn(x))

    # penultimate ReLU (1150-dim) for embedding extraction
    @property
    def penultimate(self):
        return self.MLP.MLP[5]


def count_params(model):
    cnn = sum(p.numel() for p in model.cnn.parameters() if p.requires_grad)
    mlp = sum(p.numel() for p in model.MLP.parameters() if p.requires_grad)
    return {"cnn": cnn, "mlp": mlp, "total": cnn + mlp}


if __name__ == "__main__":
    for arch in C.MODELS:
        m = Model(arch)
        print(arch, m.feat_shape, count_params(m))