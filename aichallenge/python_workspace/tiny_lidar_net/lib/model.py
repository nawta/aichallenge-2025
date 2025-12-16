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


class TinyLidarNetFusion(nn.Module):
    """
    Multi-modal CNN architecture that fuses LiDAR data with kinematic state.
    
    Uses Late Fusion approach:
    - LiDAR branch: Same as TinyLidarNet (5 Conv layers)
    - State branch: Single FC layer to encode kinematic state
    - Fusion: Concatenate features and pass through FC layers
    
    Kinematic state features (13 dimensions):
    - Position (x, y, z): 3
    - Orientation quaternion (x, y, z, w): 4
    - Linear velocity (vx, vy, vz): 3
    - Angular velocity (wx, wy, wz): 3
    
    Assumes default input_dim=1080 for shape annotations.
    """

    def __init__(
        self, 
        input_dim: int = 1080, 
        state_dim: int = 13,
        output_dim: int = 2
    ):
        super().__init__()
        
        self.state_dim = state_dim

        # --- LiDAR Branch (same as TinyLidarNet) ---
        # Input: 1080
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)  # -> 268
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)  # -> 66
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)  # -> 32
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)            # -> 30
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)            # -> 28
        
        # Calculate LiDAR flatten dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy)))))
            self.lidar_flatten_dim = out.view(1, -1).shape[1]  # 1792

        # --- State Branch ---
        self.state_fc = nn.Linear(state_dim, 64)
        
        # --- Fusion Head ---
        # Concat: lidar_features (1792) + state_features (64) = 1856
        fusion_dim = self.lidar_flatten_dim + 64
        
        self.fc1 = nn.Linear(fusion_dim, 100)
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
        lidar: Float[Tensor, "batch 1 1080"],
        state: Float[Tensor, "batch 13"]
    ) -> Float[Tensor, "batch 2"]:
        """
        Forward pass with LiDAR and kinematic state inputs.
        
        Args:
            lidar: Normalized LiDAR scan data, shape (batch, 1, input_dim)
            state: Normalized kinematic state, shape (batch, state_dim)
            
        Returns:
            Control output [acceleration, steering], shape (batch, output_dim)
        """
        # --- LiDAR Branch ---
        x = F.relu(self.conv1(lidar))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # Flatten LiDAR features: (Batch, 64, 28) -> (Batch, 1792)
        lidar_features = torch.flatten(x, start_dim=1)
        
        # --- State Branch ---
        state_features = F.relu(self.state_fc(state))  # (Batch, 64)
        
        # --- Late Fusion ---
        fused = torch.cat([lidar_features, state_features], dim=1)  # (Batch, 1856)
        
        # --- Regression Head ---
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Output Layer
        x = torch.tanh(self.fc4(x))
        
        return x


# =============================================================================
# Temporal Models
# =============================================================================

class TinyLidarNetStacked(nn.Module):
    """
    Frame Stacking temporal model.
    
    Stacks multiple consecutive LiDAR frames as input channels.
    Simple baseline for temporal modeling.
    
    Args:
        input_dim: LiDAR scan dimension (default: 1080)
        state_dim: Kinematic state dimension (default: 13)
        seq_len: Number of frames to stack (default: 10)
        output_dim: Output dimension (default: 2)
    """

    def __init__(
        self, 
        input_dim: int = 1080, 
        state_dim: int = 13,
        seq_len: int = 10,
        output_dim: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.seq_len = seq_len

        # --- LiDAR Branch (input channels = seq_len) ---
        self.conv1 = nn.Conv1d(seq_len, 24, kernel_size=10, stride=4)  # -> 268
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)        # -> 66
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)        # -> 32
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)                  # -> 30
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)                  # -> 28
        
        # Calculate flatten dimension
        with torch.no_grad():
            dummy = torch.zeros(1, seq_len, input_dim)
            out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy)))))
            self.lidar_flatten_dim = out.view(1, -1).shape[1]

        # --- State Branch (process flattened odom sequence) ---
        self.state_fc = nn.Linear(state_dim * seq_len, 64)
        
        # --- Fusion Head ---
        fusion_dim = self.lidar_flatten_dim + 64
        
        self.fc1 = nn.Linear(fusion_dim, 100)
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
        scans: Tensor,
        odoms: Tensor
    ) -> Tensor:
        """
        Forward pass with stacked LiDAR frames and odom sequence.
        
        Args:
            scans: Stacked LiDAR scans, shape (batch, seq_len, scan_dim)
            odoms: Odom sequence, shape (batch, seq_len, state_dim)
            
        Returns:
            Control output [acceleration, steering], shape (batch, output_dim)
        """
        # --- LiDAR Branch ---
        # scans is already (batch, seq_len, scan_dim), use as (batch, channels, length)
        x = F.relu(self.conv1(scans))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        lidar_features = torch.flatten(x, start_dim=1)
        
        # --- State Branch ---
        # Flatten odom sequence: (batch, seq_len, state_dim) -> (batch, seq_len * state_dim)
        odom_flat = odoms.view(odoms.size(0), -1)
        state_features = F.relu(self.state_fc(odom_flat))
        
        # --- Fusion ---
        fused = torch.cat([lidar_features, state_features], dim=1)
        
        # --- Output Head ---
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return torch.tanh(self.fc4(x))


class TinyLidarNetBiLSTM(nn.Module):
    """
    Bidirectional LSTM temporal model.
    
    - Training: Uses bidirectional LSTM (sees future frames)
    - Inference: Uses forward LSTM only with projection layer
    
    Args:
        input_dim: LiDAR scan dimension (default: 1080)
        state_dim: Kinematic state dimension (default: 13)
        hidden_size: LSTM hidden size (default: 128)
        output_dim: Output dimension (default: 2)
    """

    def __init__(
        self, 
        input_dim: int = 1080, 
        state_dim: int = 13,
        hidden_size: int = 128,
        output_dim: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size

        # --- Shared CNN Encoder (same as TinyLidarNet) ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy)))))
            self.lidar_flatten_dim = out.view(1, -1).shape[1]  # 1792

        # --- State Encoder ---
        self.state_fc = nn.Linear(state_dim, 64)
        
        # --- Feature Projection ---
        # 1792 + 64 = 1856 -> hidden_size
        self.feature_proj = nn.Linear(self.lidar_flatten_dim + 64, hidden_size)
        
        # --- Bidirectional LSTM ---
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # --- Forward-only projection (for inference) ---
        # Maps forward LSTM output (hidden_size) to match bidirectional output (hidden_size * 2)
        self.forward_proj = nn.Linear(hidden_size, hidden_size * 2)
        
        # --- Output Head (takes 256-dim input from BiLSTM) ---
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def encode_frame(self, scan: Tensor, odom: Tensor) -> Tensor:
        """Encode a single frame (scan + odom) into feature vector."""
        # scan: (batch, 1, scan_dim)
        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        scan_feat = torch.flatten(x, start_dim=1)  # (batch, 1792)
        
        odom_feat = F.relu(self.state_fc(odom))  # (batch, 64)
        
        combined = torch.cat([scan_feat, odom_feat], dim=1)  # (batch, 1856)
        return F.relu(self.feature_proj(combined))  # (batch, hidden_size)

    def forward(
        self, 
        scans: Tensor,
        odoms: Tensor,
        use_bidirectional: bool = True
    ) -> Tensor:
        """
        Forward pass with sequence of LiDAR frames and odom.
        
        Args:
            scans: LiDAR sequence, shape (batch, seq_len, scan_dim)
            odoms: Odom sequence, shape (batch, seq_len, state_dim)
            use_bidirectional: If True, use BiLSTM (training). If False, use forward only (inference).
            
        Returns:
            Control output [acceleration, steering], shape (batch, output_dim)
        """
        batch_size, seq_len, _ = scans.shape
        
        # Encode each frame
        features = []
        for t in range(seq_len):
            scan_t = scans[:, t:t+1, :].transpose(1, 2)  # (batch, 1, scan_dim)
            odom_t = odoms[:, t, :]  # (batch, state_dim)
            feat_t = self.encode_frame(scan_t, odom_t)  # (batch, hidden_size)
            features.append(feat_t)
        
        features = torch.stack(features, dim=1)  # (batch, seq_len, hidden_size)
        
        if use_bidirectional:
            # Training: use full BiLSTM
            lstm_out, _ = self.lstm(features)  # (batch, seq_len, hidden_size * 2)
            out = lstm_out[:, -1, :]  # Last timestep (batch, 256)
        else:
            # Inference: use forward LSTM only
            # Split BiLSTM weights and use only forward direction
            lstm_out, _ = self.lstm(features)
            # Take only forward direction output (first hidden_size dims)
            forward_out = lstm_out[:, -1, :self.hidden_size]  # (batch, hidden_size)
            out = self.forward_proj(forward_out)  # (batch, hidden_size * 2)
        
        # Output head
        out = F.relu(self.fc1(out))
        return torch.tanh(self.fc2(out))


class TemporalBlock(nn.Module):
    """
    Single TCN block with dilated causal convolution and residual connection.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        dilation: Dilation factor
        causal: If True, use causal convolution (no future info)
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        dilation: int,
        causal: bool = True
    ):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding
        if causal:
            # Causal: pad only on the left
            self.padding = (kernel_size - 1) * dilation
        else:
            # Non-causal: pad both sides
            self.padding = ((kernel_size - 1) * dilation) // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, 
            padding=0 if causal else self.padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation,
            padding=0 if causal else self.padding
        )
        
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, channels, seq_len)
            
        Returns:
            Output tensor, shape (batch, out_channels, seq_len)
        """
        if self.causal:
            # Manual causal padding (left only)
            x_padded = F.pad(x, (self.padding, 0))
            out = self.relu(self.conv1(x_padded))
            out = F.pad(out, (self.padding, 0))
            out = self.relu(self.conv2(out))
        else:
            out = self.relu(self.conv1(x))
            out = self.relu(self.conv2(out))
        
        # Residual connection
        res = self.residual(x) if self.residual else x
        return out + res


class TinyLidarNetTCN(nn.Module):
    """
    Temporal Convolutional Network model.
    
    Uses dilated convolutions over time for temporal modeling.
    
    Args:
        input_dim: LiDAR scan dimension (default: 1080)
        state_dim: Kinematic state dimension (default: 13)
        hidden_size: TCN hidden size (default: 128)
        num_levels: Number of TCN blocks with increasing dilation (default: 3)
        kernel_size: Convolution kernel size (default: 3)
        causal: If True, use causal convolution (default: True for inference compatibility)
        output_dim: Output dimension (default: 2)
    """

    def __init__(
        self, 
        input_dim: int = 1080, 
        state_dim: int = 13,
        hidden_size: int = 128,
        num_levels: int = 3,
        kernel_size: int = 3,
        causal: bool = True,
        output_dim: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.causal = causal

        # --- Shared CNN Encoder ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy)))))
            self.lidar_flatten_dim = out.view(1, -1).shape[1]

        # --- State Encoder ---
        self.state_fc = nn.Linear(state_dim, 64)
        
        # --- Feature Projection ---
        self.feature_proj = nn.Linear(self.lidar_flatten_dim + 64, hidden_size)
        
        # --- TCN Blocks with increasing dilation ---
        self.tcn_blocks = nn.ModuleList([
            TemporalBlock(hidden_size, hidden_size, kernel_size, dilation=2**i, causal=causal)
            for i in range(num_levels)  # dilation: 1, 2, 4
        ])
        
        # --- Output Head ---
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode_frame(self, scan: Tensor, odom: Tensor) -> Tensor:
        """Encode a single frame (scan + odom) into feature vector."""
        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        scan_feat = torch.flatten(x, start_dim=1)
        
        odom_feat = F.relu(self.state_fc(odom))
        
        combined = torch.cat([scan_feat, odom_feat], dim=1)
        return F.relu(self.feature_proj(combined))

    def forward(
        self, 
        scans: Tensor,
        odoms: Tensor
    ) -> Tensor:
        """
        Forward pass with sequence of LiDAR frames and odom.
        
        Args:
            scans: LiDAR sequence, shape (batch, seq_len, scan_dim)
            odoms: Odom sequence, shape (batch, seq_len, state_dim)
            
        Returns:
            Control output [acceleration, steering], shape (batch, output_dim)
        """
        batch_size, seq_len, _ = scans.shape
        
        # Encode each frame
        features = []
        for t in range(seq_len):
            scan_t = scans[:, t:t+1, :].transpose(1, 2)  # (batch, 1, scan_dim)
            odom_t = odoms[:, t, :]  # (batch, state_dim)
            feat_t = self.encode_frame(scan_t, odom_t)  # (batch, hidden_size)
            features.append(feat_t)
        
        # Stack: (batch, seq_len, hidden) -> (batch, hidden, seq_len) for Conv1d
        x = torch.stack(features, dim=2)  # (batch, hidden_size, seq_len)
        
        # Apply TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Take last timestep output
        out = x[:, :, -1]  # (batch, hidden_size)
        
        # Output head
        out = F.relu(self.fc1(out))
        return torch.tanh(self.fc2(out))
