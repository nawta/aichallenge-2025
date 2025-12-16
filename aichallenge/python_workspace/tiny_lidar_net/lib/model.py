import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

class TinyLidarNet(nn.Module):
    """
    Standard CNN architecture for 1D LiDAR data processing.
    Assumes default input_dim=1080 for shape annotations.
    """

    def __init__(self, input_dim: int = 1080, output_dim: int = 2):
        super().__init__()

        # --- Convolutional Layers ---
        # Input: 1080
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)  # -> (1080-10)/4 + 1 = 268
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)  # -> (268-8)/4 + 1 = 66
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)  # -> (66-4)/2 + 1 = 32
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)            # -> (32-3)/1 + 1 = 30
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)            # -> (30-3)/1 + 1 = 28
        
        # Flatten size: 64 ch * 28 length = 1792
        
        # --- Fully Connected Layers ---
        # Note: Dynamic calculation is good, but for jaxtyping clarity we assume logic matches
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy)))))
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        x: Float[Tensor, "batch 1 1080"]
    ) -> Float[Tensor, "batch 2"]:
        
        # Feature Extraction (Conv + ReLU)
        # Input: [B, 1, 1080]
        x: Float[Tensor, "batch 24 268"] = F.relu(self.conv1(x))
        x: Float[Tensor, "batch 36 66"]  = F.relu(self.conv2(x))
        x: Float[Tensor, "batch 48 32"]  = F.relu(self.conv3(x))
        x: Float[Tensor, "batch 64 30"]  = F.relu(self.conv4(x))
        x: Float[Tensor, "batch 64 28"]  = F.relu(self.conv5(x))

        # Flatten: (Batch, 64, 28) -> (Batch, 1792)
        x: Float[Tensor, "batch 1792"] = torch.flatten(x, start_dim=1)

        # Regression Head (FC + ReLU)
        x: Float[Tensor, "batch 100"] = F.relu(self.fc1(x))
        x: Float[Tensor, "batch 50"]  = F.relu(self.fc2(x))
        x: Float[Tensor, "batch 10"]  = F.relu(self.fc3(x))

        # Output Layer
        x: Float[Tensor, "batch 2"] = torch.tanh(self.fc4(x))
        
        return x


class TinyLidarNetSmall(nn.Module):
    """
    Lightweight CNN architecture.
    Assumes default input_dim=1080 for shape annotations.
    """

    def __init__(self, input_dim: int = 1080, output_dim: int = 2):
        super().__init__()

        # --- Convolutional Layers ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4) # -> 268
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4) # -> 66
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2) # -> 32
        
        # Flatten size: 48 ch * 32 length = 1536

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv3(self.conv2(self.conv1(dummy)))
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        x: Float[Tensor, "batch 1 1080"]
    ) -> Float[Tensor, "batch 2"]:
        
        # Feature Extraction
        x: Float[Tensor, "batch 24 268"] = F.relu(self.conv1(x))
        x: Float[Tensor, "batch 36 66"]  = F.relu(self.conv2(x))
        x: Float[Tensor, "batch 48 32"]  = F.relu(self.conv3(x))
        
        # Flatten: (Batch, 48, 32) -> (Batch, 1536)
        x: Float[Tensor, "batch 1536"] = torch.flatten(x, start_dim=1)
        
        # Regression Head
        x: Float[Tensor, "batch 100"] = F.relu(self.fc1(x))
        x: Float[Tensor, "batch 50"]  = F.relu(self.fc2(x))
        
        # Output Layer
        x: Float[Tensor, "batch 2"] = torch.tanh(self.fc3(x))
        
        return x


class ResidualBlock1d(nn.Module):
    """
    1D Residual Block with skip connection.
    Helps training deeper networks by mitigating vanishing gradient problem.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)


class TinyLidarNetDeep(nn.Module):
    """
    Deep CNN architecture with Residual connections for 1D LiDAR data processing.
    
    Features:
    - 7 Convolutional layers (vs 5 in standard)
    - 2 Residual blocks for better gradient flow
    - 6 Fully connected layers (vs 4 in standard)
    - Batch Normalization for stable training
    
    Assumes default input_dim=1080 for shape annotations.
    """

    def __init__(self, input_dim: int = 1080, output_dim: int = 2):
        super().__init__()

        # --- Convolutional Layers (Downsampling) ---
        # Input: 1080
        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, stride=4)   # -> 268
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 48, kernel_size=8, stride=4)   # -> 66
        self.bn2 = nn.BatchNorm1d(48)
        
        self.conv3 = nn.Conv1d(48, 64, kernel_size=4, stride=2)   # -> 32
        self.bn3 = nn.BatchNorm1d(64)
        
        # --- Residual Blocks (preserve spatial dimension) ---
        self.res_block1 = ResidualBlock1d(64)
        self.res_block2 = ResidualBlock1d(64)
        
        # --- Additional Conv Layers ---
        self.conv4 = nn.Conv1d(64, 96, kernel_size=3)             # -> 30
        self.bn4 = nn.BatchNorm1d(96)
        
        self.conv5 = nn.Conv1d(96, 96, kernel_size=3)             # -> 28
        self.bn5 = nn.BatchNorm1d(96)
        
        # --- Calculate Flatten Dimension ---
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            x = F.relu(self.bn1(self.conv1(dummy)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.res_block1(x)
            x = self.res_block2(x)
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            self.flatten_dim = x.view(1, -1).shape[1]

        # --- Fully Connected Layers (Deeper) ---
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        self.fc6 = nn.Linear(10, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        x: Float[Tensor, "batch 1 1080"]
    ) -> Float[Tensor, "batch 2"]:
        
        # Feature Extraction with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Residual Blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Additional Conv Layers
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Deeper Regression Head with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        # Output Layer
        x = torch.tanh(self.fc6(x))
        
        return x
