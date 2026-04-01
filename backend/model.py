import torch
import torch.nn as nn
import torch.nn.functional as F


class TextSimilarityCNNLegacy(nn.Module):
    """Original Conv1D architecture (trained before the Conv2D refactor).

    Kept for backwards-compatibility so that checkpoints trained with the old
    architecture can still be loaded and used for inference without re-training.

    Input shape: [batch, input_dim]  — flat concatenation of all feature maps.
    """

    def __init__(self, input_dim: int = 57344, hidden_dim: int = 128):
        super().__init__()
        self.fc_reduce1 = nn.Linear(input_dim, 4096)
        self.fc_reduce2 = nn.Linear(4096, 1024)
        self.fc_reduce3 = nn.Linear(1024, 256)
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((hidden_dim // 2) * 64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(4096)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc_reduce1(x)); x = self.batch_norm1(x); x = self.dropout(x)
        x = F.relu(self.fc_reduce2(x)); x = self.batch_norm2(x); x = self.dropout(x)
        x = F.relu(self.fc_reduce3(x)); x = self.batch_norm3(x); x = self.dropout(x)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x)); x = F.max_pool1d(x, 2, 2); x = self.dropout(x)
        x = F.relu(self.conv2(x)); x = F.max_pool1d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.batch_norm4(x); x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class TextSimilarityCNN(nn.Module):
    """CNN that operates on stacked 2D feature maps from the sentence-pair pipeline.

    Input shape: [batch, num_features, spatial_size, spatial_size]
        num_features  — one channel per feature map (semantic, lexical, NLI, entity …)
        spatial_size  — side length of each resized sentence-pair similarity matrix
                        (default 32 — set by TARGET_SIZE in Postprocess/__addpad.py)

    Feature maps are produced by resize_matrix() which uses bilinear interpolation to
    map the raw n×m cross-sentence matrix to a fixed spatial_size×spatial_size grid.
    Every cell carries signal from actual sentence pairs — no zero-padding artefacts.
    """

    def __init__(self, num_features: int, hidden_dim: int = 256, spatial_size: int = 32):
        """
        Args:
            num_features (int):  Number of input feature-map channels.
            hidden_dim (int):    Base width for convolutional feature maps.
            spatial_size (int):  Spatial side length of the input maps (default 32).
                                 Must be divisible by 8 (three MaxPool(2) layers).
        """
        super(TextSimilarityCNN, self).__init__()

        self.num_features = num_features
        self.spatial_size = spatial_size

        # --- Spatial feature extraction ---
        # Block 1: num_features → hidden_dim,   spatial_size → spatial_size/2
        self.conv1 = nn.Conv2d(num_features, hidden_dim, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(hidden_dim)

        # Block 2: hidden_dim → hidden_dim//2,  spatial_size/2 → spatial_size/4
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(hidden_dim // 2)

        # Block 3: hidden_dim//2 → hidden_dim//4, spatial_size/4 → spatial_size/8
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(hidden_dim // 4)

        # --- Classification head ---
        # After 3× MaxPool(2,2): side = spatial_size/8
        reduced = spatial_size // 8
        flat_dim = (hidden_dim // 4) * reduced * reduced

        self.fc1      = nn.Linear(flat_dim, hidden_dim)
        self.bn_fc1   = nn.BatchNorm1d(hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, 1)

        self.dropout  = nn.Dropout(0.3)
        self.pool     = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: [batch, num_features, spatial_size, spatial_size]

        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # → [B, H,   S/2, S/2]
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # → [B, H/2, S/4, S/4]
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # → [B, H/4, S/8, S/8]
        x = self.dropout(x)

        x = x.view(x.size(0), -1)                        # → [B, flat_dim]

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))                   # → [B, 1]

        return x
