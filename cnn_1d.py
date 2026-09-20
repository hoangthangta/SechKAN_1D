import torch.nn as nn



class CNN1D_Crop(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        # Input: (B, 46) -> (B, 1, 46)
        self.conv1 = nn.Conv1d(1, 16, 80, 4, 38)
        self.bn1 = nn.BatchNorm1d(16)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 36, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(36)
        self.act2 = nn.SiLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(36, 54, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(54)
        self.act3 = nn.SiLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(54, 58)
        self.act4 = nn.SiLU()
        self.fc2 = nn.Linear(58, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)

        return x
        
class CNN1D_Ele_Dev(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Input: (B, 96)
        # -> (B, 1, 96)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=35,
            kernel_size=80,
            stride=4,
            padding=38
        )
        self.bn1 = nn.BatchNorm1d(35)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(
            in_channels=35,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.SiLU()
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=37,
            kernel_size=5,
            padding=2
        )
        self.bn3 = nn.BatchNorm1d(37)
        self.act3 = nn.SiLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(37, 24)
        self.act4 = nn.SiLU()

        self.fc2 = nn.Linear(24, num_classes)

    def forward(self, x):
        # (B, 96) -> (B, 1, 96)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)

        # (B, 37, 1) -> (B, 37)
        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.act4(x)

        x = self.fc2(x)

        return x


class CNN1D_HAR(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 30, 80, 4, 38)
        self.bn1 = nn.BatchNorm1d(30)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(30, 224, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(224)
        self.act2 = nn.SiLU()
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(224, 96, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(96)
        self.act3 = nn.SiLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(96, 24)
        self.act4 = nn.SiLU()
        self.fc2 = nn.Linear(24, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)

        return x