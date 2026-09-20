import torch
import torch.nn as nn

class DSBlock1D_Crop(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.dw = nn.Conv1d(
            in_ch, in_ch, 5, stride, 2,
            groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = nn.SiLU()

        self.pw = nn.Conv1d(
            in_ch, out_ch, 1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.SiLU()

        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv1d(
                    in_ch, out_ch, 1, stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_ch)
            )
        )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.pw(x)
        x = self.bn2(x)

        x = x + identity
        x = self.act2(x)

        return x


class DSCNN1D_Crop(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        # Input: (B, 46)

        self.conv1 = nn.Conv1d(
            1, 16, 5, 2, 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        # Stage 1
        self.block1 = DSBlock1D_Crop(16, 16)

        self.block2 = DSBlock1D_Crop(
            16, 32, stride=2
        )

        # Stage 2
        self.block3 = DSBlock1D_Crop(32, 32)

        self.block4 = DSBlock1D_Crop(
            32, 63, stride=2
        )

        # Stage 3
        self.block5 = DSBlock1D_Crop(63, 63)

        self.block6 = DSBlock1D_Crop(63, 63)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(63, 24)
        self.act2 = nn.SiLU()

        self.fc2 = nn.Linear(
            24, num_classes
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.pool(x)
        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.act2(x)
        x = self.fc2(x)

        return x

class DSBlock1D_Ele_Dev(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        # Depthwise convolution
        self.dw = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=in_ch,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = nn.SiLU()

        # Pointwise convolution
        self.pw = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.SiLU()

        # Shortcut
        if in_ch == out_ch and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.pw(x)
        x = self.bn2(x)

        x = x + identity
        x = self.act2(x)

        return x


class DSCNN1D_Ele_Dev(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Input: (B, 96)
        # -> (B, 1, 96)

        self.conv1 = nn.Conv1d(
            1, 16,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        # Stage 1
        self.block1 = DSBlock1D_Ele_Dev(16, 16)

        self.block2 = DSBlock1D_Ele_Dev(
            16, 40, stride=2
        )

        # Stage 2
        self.block3 = DSBlock1D_Ele_Dev(40, 40)

        self.block4 = DSBlock1D_Ele_Dev(
            40, 70, stride=2
        )

        # Stage 3
        self.block5 = DSBlock1D_Ele_Dev(70, 70)

        self.block6 = DSBlock1D_Ele_Dev(70, 70)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(70, 78)
        self.act2 = nn.SiLU()

        self.fc2 = nn.Linear(78, num_classes)

    def forward(self, x):
        # (B, 96) -> (B, 1, 96)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.pool(x)
        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.act2(x)

        x = self.fc2(x)

        return x


class DSBlock1D_HAR(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.dw = nn.Conv1d(in_ch, in_ch, 5, stride, 2, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = nn.SiLU()

        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.SiLU()

        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.act1(self.bn1(self.dw(x)))
        x = self.bn2(self.pw(x))
        return self.act2(x + identity)


class DSCNN1D_HAR(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 24, 5, 2, 2, bias=False)
        self.bn1 = nn.BatchNorm1d(24)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        self.block1 = DSBlock1D_HAR(24, 24)
        self.block2 = DSBlock1D_HAR(24, 112, stride=2)

        self.block3 = DSBlock1D_HAR(112, 112)
        self.block4 = DSBlock1D_HAR(112, 176, stride=2)

        self.block5 = DSBlock1D_HAR(176, 176)
        self.block6 = DSBlock1D_HAR(176, 176)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(176, 112)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(112, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.pool(x).squeeze(-1)

        x = self.act2(self.fc1(x))
        return self.fc2(x)
        
