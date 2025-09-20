import torch
import torch.nn as nn
import torch.nn.functional as F

class TextSimilarityCNN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(TextSimilarityCNN, self).__init__()

        # Multi-stage dimensionality reduction
        self.fc_reduce1 = nn.Linear(input_dim, 4096)
        self.fc_reduce2 = nn.Linear(4096, 1024)
        self.fc_reduce3 = nn.Linear(1024, 256)

        # CNN layers
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear((hidden_dim//2) * 64, hidden_dim)  # Adjusted for the reduced dimension
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Regularization
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(4096)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # Multi-stage dimensionality reduction
        x = F.relu(self.fc_reduce1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)

        x = F.relu(self.fc_reduce2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        x = F.relu(self.fc_reduce3(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)

        # Reshape for CNN
        x = x.unsqueeze(1)  # Add channel dimension

        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.batch_norm4(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x
