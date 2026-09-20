import torch.nn as nn


class ResBlock1D_Crop(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, 3, stride, 1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_ch, out_ch, 1, stride, bias=False
                ),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.act(x + identity)


class ResNet1D_Crop(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        # Input: (B, 46) -> (B, 1, 46)
        self.conv1 = nn.Conv1d(
            1, 7, 5, 2, 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(7)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        self.block1 = ResBlock1D_Crop(7, 7)

        self.block2 = ResBlock1D_Crop(
            7, 38, stride=2
        )

        self.block3 = ResBlock1D_Crop(38, 38)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(38, 63)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(63, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x).squeeze(-1)

        x = self.act2(self.fc1(x))
        return self.fc2(x)

class ResBlock1D_Ele_Dev(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch or stride != 1:
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
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)

        return out


class ResNet1D_Ele_Dev(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Input: (B, 96)
        # -> (B, 1, 96)

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        # Residual block 1
        self.block1 = ResBlock1D_Ele_Dev(
            32,
            32
        )

        # Residual block 2: increase feature width
        self.block2 = ResBlock1D_Ele_Dev(
            32,
            40,
            stride=2
        )

        # Residual block 3
        self.block3 = ResBlock1D_Ele_Dev(
            40,
            40
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(
            40,
            19
        )
        self.act2 = nn.SiLU()

        self.fc2 = nn.Linear(
            19,
            num_classes
        )

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

        x = self.pool(x)
        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.act2(x)

        x = self.fc2(x)

        return x

    
class ResBlock1D_HAR(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.act(x + identity)


class ResNet1D_HAR(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 77, 5, 2, 2, bias=False)
        self.bn1 = nn.BatchNorm1d(77)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        self.block1 = ResBlock1D_HAR(77, 77)
        self.block2 = ResBlock1D_HAR(77, 91, stride=2)
        self.block3 = ResBlock1D_HAR(91, 91)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(91, 73)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(73, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x).squeeze(-1)

        x = self.act2(self.fc1(x))

        return self.fc2(x)
        
