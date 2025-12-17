import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these are imported from your local module as per the original code
from . import (
    conv1d,
    conv1d_padded,
    conv2d,
    conv2d_padded,
    max_pool2d,
    linear,
    relu,
    tanh,
    flatten,
    batch_norm1d,
    batch_norm2d,
    adaptive_avg_pool2d,
    kaiming_normal_init,
    zeros_init,
    ones_init,
)

# ============================================================
# PyTorch Models
# ============================================================

class TinyLidarNet(nn.Module):
    """Standard CNN model for LiDAR data (Conv5 + FC4).

    This model processes 1D LiDAR scan data through 5 convolutional layers
    followed by 4 fully connected layers.

    Attributes:
        conv1 (nn.Conv1d): First convolutional layer.
        conv2 (nn.Conv1d): Second convolutional layer.
        conv3 (nn.Conv1d): Third convolutional layer.
        conv4 (nn.Conv1d): Fourth convolutional layer.
        conv5 (nn.Conv1d): Fifth convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Output fully connected layer.
    """

    def __init__(self, input_dim=1080, output_dim=2):
        """Initializes TinyLidarNet.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        super().__init__()

        # --- Convolutional Layers ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)

        # --- Calculate Flatten Dimension ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            flatten_dim = x.view(1, -1).shape[1]

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (He) initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim) with Tanh activation.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class TinyLidarNetSmall(nn.Module):
    """Lightweight CNN model for LiDAR data (Conv3 + FC3).

    This model is a smaller version of TinyLidarNet, processing data through
    3 convolutional layers and 3 fully connected layers.
    """

    def __init__(self, input_dim=1080, output_dim=2):
        """Initializes TinyLidarNetSmall.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        super().__init__()

        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv3(self.conv2(self.conv1(dummy_input)))
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (He) initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim) with Tanh activation.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


# ============================================================
# NumPy Inference Models (Exact Naming Match with PyTorch)
# ============================================================

class TinyLidarNetNp:
    """NumPy implementation of TinyLidarNet (Conv5 + FC4).

    This class provides a pure NumPy inference implementation that matches the
    architecture of the PyTorch `TinyLidarNet` class.

    Attributes:
        params (dict): Stores weights and biases for all layers.
        strides (dict): Stores stride values for convolutional layers.
        shapes (dict): Stores parameter shapes for initialization.
    """

    def __init__(self, input_dim=1080, output_dim=2):
        """Initializes TinyLidarNetNp.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}

        # Stride definitions
        self.strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}

        # Shape definitions matching PyTorch
        self.shapes = {
            'conv1_weight': (24, 1, 10),  'conv1_bias': (24,),
            'conv2_weight': (36, 24, 8),  'conv2_bias': (36,),
            'conv3_weight': (48, 36, 4),  'conv3_bias': (48,),
            'conv4_weight': (64, 48, 3),  'conv4_bias': (64,),
            'conv5_weight': (64, 64, 3),  'conv5_bias': (64,),
        }

        flatten_dim = self._get_conv_output_dim()
        self.shapes.update({
            'fc1_weight': (100, flatten_dim), 'fc1_bias': (100,),
            'fc2_weight': (50, 100),          'fc2_bias': (50,),
            'fc3_weight': (10, 50),           'fc3_bias': (10,),
            'fc4_weight': (output_dim, 10),   'fc4_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_conv_output_dim(self):
        """Calculates the flattened dimension after the last convolution layer."""
        l = self.input_dim
        for i in range(1, 6):
            k = self.shapes[f'conv{i}_weight'][2]
            s = self.strides[f'conv{i}']
            l = (l - k) // s + 1
        c = self.shapes['conv5_weight'][0]
        return c * l

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (fan_out) and biases to zero."""
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                fan_out = shape[0] * (shape[2] if 'conv' in name else 1)
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def __call__(self, x):
        """Performs the forward pass of the model.

        Args:
            x (np.ndarray): Input array of shape (batch_size, 1, input_dim).

        Returns:
            np.ndarray: Output array of shape (batch_size, output_dim).
        """
        x = relu(conv1d(x, self.params['conv1_weight'], self.params['conv1_bias'], self.strides['conv1']))
        x = relu(conv1d(x, self.params['conv2_weight'], self.params['conv2_bias'], self.strides['conv2']))
        x = relu(conv1d(x, self.params['conv3_weight'], self.params['conv3_bias'], self.strides['conv3']))
        x = relu(conv1d(x, self.params['conv4_weight'], self.params['conv4_bias'], self.strides['conv4']))
        x = relu(conv1d(x, self.params['conv5_weight'], self.params['conv5_bias'], self.strides['conv5']))
        x = flatten(x)
        x = relu(linear(x, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        x = relu(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))
        return tanh(linear(x, self.params['fc4_weight'], self.params['fc4_bias']))


class TinyLidarNetSmallNp:
    """NumPy implementation of TinyLidarNetSmall (Conv3 + FC3).

    This class provides a pure NumPy inference implementation that matches the
    architecture of the PyTorch `TinyLidarNetSmall` class.
    """

    def __init__(self, input_dim=1080, output_dim=2):
        """Initializes TinyLidarNetSmallNp.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.strides = {'conv1': 4, 'conv2': 4, 'conv3': 2}

        self.shapes = {
            'conv1_weight': (24, 1, 10),  'conv1_bias': (24,),
            'conv2_weight': (36, 24, 8),  'conv2_bias': (36,),
            'conv3_weight': (48, 36, 4),  'conv3_bias': (48,),
        }

        flatten_dim = self._get_conv_output_dim()
        self.shapes.update({
            'fc1_weight': (100, flatten_dim), 'fc1_bias': (100,),
            'fc2_weight': (50, 100),          'fc2_bias': (50,),
            'fc3_weight': (output_dim, 50),   'fc3_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_conv_output_dim(self):
        """Calculates the flattened dimension after the last convolution layer."""
        l = self.input_dim
        for i in range(1, 4):
            k = self.shapes[f'conv{i}_weight'][2]
            s = self.strides[f'conv{i}']
            l = (l - k) // s + 1
        c = self.shapes['conv3_weight'][0]
        return c * l

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (fan_out) and biases to zero."""
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                fan_out = shape[0] * (shape[2] if 'conv' in name else 1)
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def __call__(self, x):
        """Performs the forward pass of the model.

        Args:
            x (np.ndarray): Input array of shape (batch_size, 1, input_dim).

        Returns:
            np.ndarray: Output array of shape (batch_size, output_dim).
        """
        x = relu(conv1d(x, self.params['conv1_weight'], self.params['conv1_bias'], self.strides['conv1']))
        x = relu(conv1d(x, self.params['conv2_weight'], self.params['conv2_bias'], self.strides['conv2']))
        x = relu(conv1d(x, self.params['conv3_weight'], self.params['conv3_bias'], self.strides['conv3']))
        x = flatten(x)
        x = relu(linear(x, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        return tanh(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))


class TinyLidarNetDeep(nn.Module):
    """Deep CNN model for LiDAR data with Residual connections.

    Features:
    - 5 Convolutional layers with BatchNorm
    - 2 Residual blocks for better gradient flow  
    - 6 Fully connected layers
    - Dropout for regularization

    This model is deeper than the standard TinyLidarNet and includes
    modern techniques like residual connections and batch normalization.
    """

    def __init__(self, input_dim=1080, output_dim=2):
        """Initializes TinyLidarNetDeep.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        super().__init__()

        # --- Convolutional Layers with BatchNorm ---
        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, stride=4)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 48, kernel_size=8, stride=4)
        self.bn2 = nn.BatchNorm1d(48)
        
        self.conv3 = nn.Conv1d(48, 64, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        # --- Residual Blocks ---
        self.res1_conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.res1_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.res2_conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.res2_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        
        # --- Additional Conv Layers ---
        self.conv4 = nn.Conv1d(64, 96, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(96)
        
        self.conv5 = nn.Conv1d(96, 96, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(96)

        # --- Calculate Flatten Dimension ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            # Residual block 1
            residual = x
            x = F.relu(self.res1_conv1(x))
            x = F.relu(self.res1_conv2(x) + residual)
            # Residual block 2
            residual = x
            x = F.relu(self.res2_conv1(x))
            x = F.relu(self.res2_conv2(x) + residual)
            # Additional conv layers
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            flatten_dim = x.view(1, -1).shape[1]

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        self.fc6 = nn.Linear(10, output_dim)
        
        self.dropout = nn.Dropout(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (He) initialization."""
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

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim) with Tanh activation.
        """
        # Feature extraction with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Residual block 1
        residual = x
        x = F.relu(self.res1_conv1(x))
        x = F.relu(self.res1_conv2(x) + residual)
        
        # Residual block 2
        residual = x
        x = F.relu(self.res2_conv1(x))
        x = F.relu(self.res2_conv2(x) + residual)
        
        # Additional conv layers
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        return torch.tanh(self.fc6(x))


class TinyLidarNetDeepNp:
    """NumPy implementation of TinyLidarNetDeep.

    Deep CNN architecture with Residual connections for 1D LiDAR data processing.
    This class provides a pure NumPy inference implementation that matches the
    architecture of the PyTorch `TinyLidarNetDeep` class.

    Features:
    - 5 Convolutional layers with BatchNorm
    - 2 Residual blocks for better gradient flow
    - 6 Fully connected layers
    """

    def __init__(self, input_dim=1080, output_dim=2):
        """Initializes TinyLidarNetDeepNp.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}

        # Stride definitions for main conv layers
        self.strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}

        # Shape definitions matching PyTorch
        self.shapes = {
            # Main conv layers
            'conv1_weight': (32, 1, 10),  'conv1_bias': (32,),
            'bn1_weight': (32,), 'bn1_bias': (32,), 'bn1_running_mean': (32,), 'bn1_running_var': (32,),
            
            'conv2_weight': (48, 32, 8),  'conv2_bias': (48,),
            'bn2_weight': (48,), 'bn2_bias': (48,), 'bn2_running_mean': (48,), 'bn2_running_var': (48,),
            
            'conv3_weight': (64, 48, 4),  'conv3_bias': (64,),
            'bn3_weight': (64,), 'bn3_bias': (64,), 'bn3_running_mean': (64,), 'bn3_running_var': (64,),
            
            # Residual block 1
            'res1_conv1_weight': (64, 64, 3), 'res1_conv1_bias': (64,),
            'res1_conv2_weight': (64, 64, 3), 'res1_conv2_bias': (64,),
            
            # Residual block 2
            'res2_conv1_weight': (64, 64, 3), 'res2_conv1_bias': (64,),
            'res2_conv2_weight': (64, 64, 3), 'res2_conv2_bias': (64,),
            
            # Additional conv layers
            'conv4_weight': (96, 64, 3),  'conv4_bias': (96,),
            'bn4_weight': (96,), 'bn4_bias': (96,), 'bn4_running_mean': (96,), 'bn4_running_var': (96,),
            
            'conv5_weight': (96, 96, 3),  'conv5_bias': (96,),
            'bn5_weight': (96,), 'bn5_bias': (96,), 'bn5_running_mean': (96,), 'bn5_running_var': (96,),
        }

        flatten_dim = self._get_conv_output_dim()
        self.shapes.update({
            'fc1_weight': (256, flatten_dim), 'fc1_bias': (256,),
            'fc2_weight': (128, 256),         'fc2_bias': (128,),
            'fc3_weight': (64, 128),          'fc3_bias': (64,),
            'fc4_weight': (32, 64),           'fc4_bias': (32,),
            'fc5_weight': (10, 32),           'fc5_bias': (10,),
            'fc6_weight': (output_dim, 10),   'fc6_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_conv_output_dim(self):
        """Calculates the flattened dimension after all convolution layers."""
        # Input: 1080
        l = self.input_dim
        
        # conv1: stride=4, kernel=10
        l = (l - 10) // 4 + 1  # 268
        # conv2: stride=4, kernel=8
        l = (l - 8) // 4 + 1   # 66
        # conv3: stride=2, kernel=4
        l = (l - 4) // 2 + 1   # 32
        
        # Residual blocks preserve dimension (padding=1, kernel=3)
        # l stays 32
        
        # conv4: stride=1, kernel=3
        l = (l - 3) // 1 + 1   # 30
        # conv5: stride=1, kernel=3
        l = (l - 3) // 1 + 1   # 28
        
        c = 96  # Final channel count
        return c * l

    def _initialize_weights(self):
        """Initializes weights using appropriate initialization schemes."""
        for name, shape in self.shapes.items():
            if name.endswith('_weight') and 'bn' not in name:
                # Conv and FC weights: Kaiming Normal
                fan_out = shape[0] * (shape[2] if len(shape) > 2 else 1)
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias') and 'bn' not in name:
                # Conv and FC biases: zeros
                self.params[name] = zeros_init(shape)
            elif 'bn' in name and name.endswith('_weight'):
                # BatchNorm gamma: ones
                self.params[name] = ones_init(shape)
            elif 'bn' in name and name.endswith('_bias'):
                # BatchNorm beta: zeros
                self.params[name] = zeros_init(shape)
            elif 'running_mean' in name:
                # BatchNorm running mean: zeros
                self.params[name] = zeros_init(shape)
            elif 'running_var' in name:
                # BatchNorm running var: ones
                self.params[name] = ones_init(shape)

    def __call__(self, x):
        """Performs the forward pass of the model.

        Args:
            x (np.ndarray): Input array of shape (batch_size, 1, input_dim).

        Returns:
            np.ndarray: Output array of shape (batch_size, output_dim).
        """
        # Feature extraction with BatchNorm
        x = conv1d(x, self.params['conv1_weight'], self.params['conv1_bias'], self.strides['conv1'])
        x = batch_norm1d(x, self.params['bn1_weight'], self.params['bn1_bias'], 
                         self.params['bn1_running_mean'], self.params['bn1_running_var'])
        x = relu(x)
        
        x = conv1d(x, self.params['conv2_weight'], self.params['conv2_bias'], self.strides['conv2'])
        x = batch_norm1d(x, self.params['bn2_weight'], self.params['bn2_bias'],
                         self.params['bn2_running_mean'], self.params['bn2_running_var'])
        x = relu(x)
        
        x = conv1d(x, self.params['conv3_weight'], self.params['conv3_bias'], self.strides['conv3'])
        x = batch_norm1d(x, self.params['bn3_weight'], self.params['bn3_bias'],
                         self.params['bn3_running_mean'], self.params['bn3_running_var'])
        x = relu(x)
        
        # Residual block 1
        residual = x
        x = relu(conv1d_padded(x, self.params['res1_conv1_weight'], self.params['res1_conv1_bias'], stride=1, padding=1))
        x = conv1d_padded(x, self.params['res1_conv2_weight'], self.params['res1_conv2_bias'], stride=1, padding=1)
        x = relu(x + residual)
        
        # Residual block 2
        residual = x
        x = relu(conv1d_padded(x, self.params['res2_conv1_weight'], self.params['res2_conv1_bias'], stride=1, padding=1))
        x = conv1d_padded(x, self.params['res2_conv2_weight'], self.params['res2_conv2_bias'], stride=1, padding=1)
        x = relu(x + residual)
        
        # Additional conv layers
        x = conv1d(x, self.params['conv4_weight'], self.params['conv4_bias'], self.strides['conv4'])
        x = batch_norm1d(x, self.params['bn4_weight'], self.params['bn4_bias'],
                         self.params['bn4_running_mean'], self.params['bn4_running_var'])
        x = relu(x)
        
        x = conv1d(x, self.params['conv5_weight'], self.params['conv5_bias'], self.strides['conv5'])
        x = batch_norm1d(x, self.params['bn5_weight'], self.params['bn5_bias'],
                         self.params['bn5_running_mean'], self.params['bn5_running_var'])
        x = relu(x)
        
        # Flatten and FC layers (no dropout during inference)
        x = flatten(x)
        x = relu(linear(x, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        x = relu(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))
        x = relu(linear(x, self.params['fc4_weight'], self.params['fc4_bias']))
        x = relu(linear(x, self.params['fc5_weight'], self.params['fc5_bias']))
        
        return tanh(linear(x, self.params['fc6_weight'], self.params['fc6_bias']))


class TinyLidarNetFusion(nn.Module):
    """Multi-modal CNN model fusing LiDAR with kinematic state.

    Uses Late Fusion approach:
    - LiDAR branch: Same as TinyLidarNet (5 Conv layers)
    - State branch: Single FC layer to encode kinematic state
    - Fusion: Concatenate features and pass through FC layers
    """

    def __init__(self, input_dim=1080, state_dim=13, output_dim=2):
        """Initializes TinyLidarNetFusion.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            state_dim (int): The size of the kinematic state vector. Defaults to 13.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        super().__init__()
        
        self.state_dim = state_dim

        # --- LiDAR Branch (same as TinyLidarNet) ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            lidar_flatten_dim = x.view(1, -1).shape[1]

        # --- State Branch ---
        self.state_fc = nn.Linear(state_dim, 64)
        
        # --- Fusion Head ---
        fusion_dim = lidar_flatten_dim + 64  # 1792 + 64 = 1856
        
        self.fc1 = nn.Linear(fusion_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (He) initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, lidar, state):
        """Forward pass with LiDAR and kinematic state inputs.

        Args:
            lidar (torch.Tensor): Input tensor of shape (batch_size, 1, input_dim).
            state (torch.Tensor): Input tensor of shape (batch_size, state_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim) with Tanh activation.
        """
        # LiDAR branch
        x = F.relu(self.conv1(lidar))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        lidar_features = x.view(x.size(0), -1)
        
        # State branch
        state_features = F.relu(self.state_fc(state))
        
        # Late fusion
        fused = torch.cat([lidar_features, state_features], dim=1)
        
        # Regression head
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class TinyLidarNetFusionNp:
    """NumPy implementation of TinyLidarNetFusion.

    Multi-modal CNN architecture that fuses LiDAR data with kinematic state.
    Uses Late Fusion approach for pure NumPy inference.
    """

    def __init__(self, input_dim=1080, state_dim=13, output_dim=2):
        """Initializes TinyLidarNetFusionNp.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            state_dim (int): The size of the kinematic state vector. Defaults to 13.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.params = {}

        # Stride definitions for LiDAR conv layers
        self.strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}

        # LiDAR branch shapes (same as TinyLidarNet)
        self.shapes = {
            'conv1_weight': (24, 1, 10),  'conv1_bias': (24,),
            'conv2_weight': (36, 24, 8),  'conv2_bias': (36,),
            'conv3_weight': (48, 36, 4),  'conv3_bias': (48,),
            'conv4_weight': (64, 48, 3),  'conv4_bias': (64,),
            'conv5_weight': (64, 64, 3),  'conv5_bias': (64,),
        }

        # State branch
        self.shapes.update({
            'state_fc_weight': (64, state_dim), 'state_fc_bias': (64,),
        })

        # Fusion head
        lidar_flatten_dim = self._get_lidar_output_dim()
        fusion_dim = lidar_flatten_dim + 64  # 1792 + 64 = 1856
        
        self.shapes.update({
            'fc1_weight': (100, fusion_dim), 'fc1_bias': (100,),
            'fc2_weight': (50, 100),         'fc2_bias': (50,),
            'fc3_weight': (10, 50),          'fc3_bias': (10,),
            'fc4_weight': (output_dim, 10),  'fc4_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_lidar_output_dim(self):
        """Calculates the flattened dimension after LiDAR conv layers."""
        l = self.input_dim
        for i in range(1, 6):
            k = self.shapes[f'conv{i}_weight'][2]
            s = self.strides[f'conv{i}']
            l = (l - k) // s + 1
        c = self.shapes['conv5_weight'][0]
        return c * l  # 64 * 28 = 1792

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (fan_out) and biases to zero."""
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                fan_out = shape[0] * (shape[2] if len(shape) > 2 else 1)
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def __call__(self, lidar, state):
        """Performs the forward pass of the model.

        Args:
            lidar (np.ndarray): Input array of shape (batch_size, 1, input_dim).
            state (np.ndarray): Input array of shape (batch_size, state_dim).

        Returns:
            np.ndarray: Output array of shape (batch_size, output_dim).
        """
        import numpy as np
        
        # LiDAR branch
        x = relu(conv1d(lidar, self.params['conv1_weight'], self.params['conv1_bias'], self.strides['conv1']))
        x = relu(conv1d(x, self.params['conv2_weight'], self.params['conv2_bias'], self.strides['conv2']))
        x = relu(conv1d(x, self.params['conv3_weight'], self.params['conv3_bias'], self.strides['conv3']))
        x = relu(conv1d(x, self.params['conv4_weight'], self.params['conv4_bias'], self.strides['conv4']))
        x = relu(conv1d(x, self.params['conv5_weight'], self.params['conv5_bias'], self.strides['conv5']))
        lidar_features = flatten(x)
        
        # State branch
        state_features = relu(linear(state, self.params['state_fc_weight'], self.params['state_fc_bias']))
        
        # Late fusion (concatenate along feature dimension)
        fused = np.concatenate([lidar_features, state_features], axis=1)
        
        # Regression head
        x = relu(linear(fused, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        x = relu(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))
        
        return tanh(linear(x, self.params['fc4_weight'], self.params['fc4_bias']))


# =============================================================================
# Temporal NumPy Models
# =============================================================================

class TinyLidarNetStackedNp:
    """NumPy implementation of TinyLidarNetStacked (Frame Stacking).

    Stacks multiple consecutive LiDAR frames as input channels.
    """

    def __init__(self, input_dim=1080, state_dim=13, seq_len=10, output_dim=2):
        """Initializes TinyLidarNetStackedNp.

        Args:
            input_dim (int): The size of the input LiDAR scan array.
            state_dim (int): The size of the kinematic state vector.
            seq_len (int): Number of frames to stack.
            output_dim (int): The size of the output prediction.
        """
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.params = {}

        # Stride definitions
        self.strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}

        # LiDAR branch (input channels = seq_len)
        self.shapes = {
            'conv1_weight': (24, seq_len, 10), 'conv1_bias': (24,),
            'conv2_weight': (36, 24, 8),       'conv2_bias': (36,),
            'conv3_weight': (48, 36, 4),       'conv3_bias': (48,),
            'conv4_weight': (64, 48, 3),       'conv4_bias': (64,),
            'conv5_weight': (64, 64, 3),       'conv5_bias': (64,),
        }

        # State branch (flattened odom sequence)
        self.shapes.update({
            'state_fc_weight': (64, state_dim * seq_len), 'state_fc_bias': (64,),
        })

        # Fusion head
        lidar_flatten_dim = self._get_lidar_output_dim()
        fusion_dim = lidar_flatten_dim + 64
        
        self.shapes.update({
            'fc1_weight': (100, fusion_dim), 'fc1_bias': (100,),
            'fc2_weight': (50, 100),         'fc2_bias': (50,),
            'fc3_weight': (10, 50),          'fc3_bias': (10,),
            'fc4_weight': (output_dim, 10),  'fc4_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_lidar_output_dim(self):
        """Calculates the flattened dimension after LiDAR conv layers."""
        l = self.input_dim
        for i in range(1, 6):
            k = self.shapes[f'conv{i}_weight'][2]
            s = self.strides[f'conv{i}']
            l = (l - k) // s + 1
        c = self.shapes['conv5_weight'][0]
        return c * l

    def _initialize_weights(self):
        """Initializes weights."""
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                fan_out = shape[0] * (shape[2] if len(shape) > 2 else 1)
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def __call__(self, scans, odoms):
        """Performs the forward pass.

        Args:
            scans (np.ndarray): Stacked scans of shape (batch, seq_len, scan_dim).
            odoms (np.ndarray): Odom sequence of shape (batch, seq_len, state_dim).

        Returns:
            np.ndarray: Output array of shape (batch, output_dim).
        """
        import numpy as np
        
        # scans: (batch, seq_len, scan_dim) - already in channel format for conv1d
        x = relu(conv1d(scans, self.params['conv1_weight'], self.params['conv1_bias'], self.strides['conv1']))
        x = relu(conv1d(x, self.params['conv2_weight'], self.params['conv2_bias'], self.strides['conv2']))
        x = relu(conv1d(x, self.params['conv3_weight'], self.params['conv3_bias'], self.strides['conv3']))
        x = relu(conv1d(x, self.params['conv4_weight'], self.params['conv4_bias'], self.strides['conv4']))
        x = relu(conv1d(x, self.params['conv5_weight'], self.params['conv5_bias'], self.strides['conv5']))
        lidar_features = flatten(x)
        
        # Flatten odom: (batch, seq_len, state_dim) -> (batch, seq_len * state_dim)
        odom_flat = odoms.reshape(odoms.shape[0], -1)
        state_features = relu(linear(odom_flat, self.params['state_fc_weight'], self.params['state_fc_bias']))
        
        # Fusion
        fused = np.concatenate([lidar_features, state_features], axis=1)
        
        # Output head
        x = relu(linear(fused, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        x = relu(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))
        
        return tanh(linear(x, self.params['fc4_weight'], self.params['fc4_bias']))


class TinyLidarNetBiLSTMNp:
    """NumPy implementation of TinyLidarNetBiLSTM.

    For inference, uses forward LSTM only with projection layer.
    Maintains hidden state for efficient sequential processing.
    """

    def __init__(self, input_dim=1080, state_dim=13, hidden_size=128, output_dim=2):
        """Initializes TinyLidarNetBiLSTMNp.

        Args:
            input_dim (int): The size of the input LiDAR scan array.
            state_dim (int): The size of the kinematic state vector.
            hidden_size (int): LSTM hidden size.
            output_dim (int): The size of the output prediction.
        """
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.params = {}
        
        # Hidden state for sequential inference
        self.h = None  # (1, hidden_size)
        self.c = None  # (1, hidden_size)

        # Stride definitions
        self.strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}

        # CNN Encoder
        self.shapes = {
            'conv1_weight': (24, 1, 10),  'conv1_bias': (24,),
            'conv2_weight': (36, 24, 8),  'conv2_bias': (36,),
            'conv3_weight': (48, 36, 4),  'conv3_bias': (48,),
            'conv4_weight': (64, 48, 3),  'conv4_bias': (64,),
            'conv5_weight': (64, 64, 3),  'conv5_bias': (64,),
        }

        # State encoder
        self.shapes.update({
            'state_fc_weight': (64, state_dim), 'state_fc_bias': (64,),
        })

        # Feature projection
        lidar_flatten_dim = self._get_lidar_output_dim()
        self.shapes.update({
            'feature_proj_weight': (hidden_size, lidar_flatten_dim + 64),
            'feature_proj_bias': (hidden_size,),
        })

        # Forward LSTM weights (from BiLSTM, using only forward direction)
        # LSTM has 4 gates: input, forget, cell, output
        self.shapes.update({
            'lstm_weight_ih_l0': (4 * hidden_size, hidden_size),  # Forward input-hidden
            'lstm_weight_hh_l0': (4 * hidden_size, hidden_size),  # Forward hidden-hidden
            'lstm_bias_ih_l0': (4 * hidden_size,),
            'lstm_bias_hh_l0': (4 * hidden_size,),
        })

        # Forward projection (maps forward LSTM output to BiLSTM output dimension)
        self.shapes.update({
            'forward_proj_weight': (hidden_size * 2, hidden_size),
            'forward_proj_bias': (hidden_size * 2,),
        })

        # Output head
        self.shapes.update({
            'fc1_weight': (64, hidden_size * 2), 'fc1_bias': (64,),
            'fc2_weight': (output_dim, 64),      'fc2_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_lidar_output_dim(self):
        """Calculates the flattened dimension after LiDAR conv layers."""
        l = self.input_dim
        for i in range(1, 6):
            k = self.shapes[f'conv{i}_weight'][2]
            s = self.strides[f'conv{i}']
            l = (l - k) // s + 1
        c = self.shapes['conv5_weight'][0]
        return c * l

    def _initialize_weights(self):
        """Initializes weights."""
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                fan_out = shape[0] * (shape[2] if len(shape) > 2 else 1)
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def reset_state(self):
        """Resets LSTM hidden state."""
        self.h = None
        self.c = None

    def _encode_frame(self, scan, odom):
        """Encode a single frame."""
        import numpy as np
        
        # scan: (1, 1, scan_dim)
        x = relu(conv1d(scan, self.params['conv1_weight'], self.params['conv1_bias'], self.strides['conv1']))
        x = relu(conv1d(x, self.params['conv2_weight'], self.params['conv2_bias'], self.strides['conv2']))
        x = relu(conv1d(x, self.params['conv3_weight'], self.params['conv3_bias'], self.strides['conv3']))
        x = relu(conv1d(x, self.params['conv4_weight'], self.params['conv4_bias'], self.strides['conv4']))
        x = relu(conv1d(x, self.params['conv5_weight'], self.params['conv5_bias'], self.strides['conv5']))
        scan_feat = flatten(x)
        
        odom_feat = relu(linear(odom, self.params['state_fc_weight'], self.params['state_fc_bias']))
        
        combined = np.concatenate([scan_feat, odom_feat], axis=1)
        return relu(linear(combined, self.params['feature_proj_weight'], self.params['feature_proj_bias']))

    def _lstm_step(self, x, h, c):
        """Single LSTM step."""
        import numpy as np
        
        # x: (batch, hidden_size), h: (batch, hidden_size), c: (batch, hidden_size)
        gates = (linear(x, self.params['lstm_weight_ih_l0'], self.params['lstm_bias_ih_l0']) +
                 linear(h, self.params['lstm_weight_hh_l0'], self.params['lstm_bias_hh_l0']))
        
        # Split into 4 gates
        hs = self.hidden_size
        i_gate = 1 / (1 + np.exp(-gates[:, :hs]))           # sigmoid
        f_gate = 1 / (1 + np.exp(-gates[:, hs:2*hs]))       # sigmoid
        g_gate = np.tanh(gates[:, 2*hs:3*hs])               # tanh
        o_gate = 1 / (1 + np.exp(-gates[:, 3*hs:]))         # sigmoid
        
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * np.tanh(c_new)
        
        return h_new, c_new

    def __call__(self, scan, odom):
        """Forward pass for single frame (sequential inference).

        Args:
            scan (np.ndarray): Single scan of shape (1, 1, scan_dim).
            odom (np.ndarray): Single odom of shape (1, state_dim).

        Returns:
            np.ndarray: Output array of shape (1, output_dim).
        """
        import numpy as np
        
        # Initialize hidden state if needed
        if self.h is None:
            self.h = np.zeros((1, self.hidden_size), dtype=np.float32)
            self.c = np.zeros((1, self.hidden_size), dtype=np.float32)
        
        # Encode frame
        features = self._encode_frame(scan, odom)  # (1, hidden_size)
        
        # Forward LSTM step
        self.h, self.c = self._lstm_step(features, self.h, self.c)
        
        # Project to match BiLSTM dimension
        projected = linear(self.h, self.params['forward_proj_weight'], self.params['forward_proj_bias'])
        
        # Output head
        x = relu(linear(projected, self.params['fc1_weight'], self.params['fc1_bias']))
        return tanh(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))


class TinyLidarNetTCNNp:
    """NumPy implementation of TinyLidarNetTCN.

    Uses causal dilated convolutions for temporal modeling.
    Requires frame buffer for inference.
    """

    def __init__(self, input_dim=1080, state_dim=13, hidden_size=128, 
                 num_levels=3, kernel_size=3, output_dim=2):
        """Initializes TinyLidarNetTCNNp.

        Args:
            input_dim (int): The size of the input LiDAR scan array.
            state_dim (int): The size of the kinematic state vector.
            hidden_size (int): TCN hidden size.
            num_levels (int): Number of TCN blocks.
            kernel_size (int): Convolution kernel size.
            output_dim (int): The size of the output prediction.
        """
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.params = {}

        # Stride definitions for CNN encoder
        self.strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}

        # CNN Encoder
        self.shapes = {
            'conv1_weight': (24, 1, 10),  'conv1_bias': (24,),
            'conv2_weight': (36, 24, 8),  'conv2_bias': (36,),
            'conv3_weight': (48, 36, 4),  'conv3_bias': (48,),
            'conv4_weight': (64, 48, 3),  'conv4_bias': (64,),
            'conv5_weight': (64, 64, 3),  'conv5_bias': (64,),
        }

        # State encoder
        self.shapes.update({
            'state_fc_weight': (64, state_dim), 'state_fc_bias': (64,),
        })

        # Feature projection
        lidar_flatten_dim = self._get_lidar_output_dim()
        self.shapes.update({
            'feature_proj_weight': (hidden_size, lidar_flatten_dim + 64),
            'feature_proj_bias': (hidden_size,),
        })

        # TCN blocks (causal dilated convolutions)
        for i in range(num_levels):
            self.shapes.update({
                f'tcn{i}_conv1_weight': (hidden_size, hidden_size, kernel_size),
                f'tcn{i}_conv1_bias': (hidden_size,),
                f'tcn{i}_conv2_weight': (hidden_size, hidden_size, kernel_size),
                f'tcn{i}_conv2_bias': (hidden_size,),
            })

        # Output head
        self.shapes.update({
            'fc1_weight': (64, hidden_size), 'fc1_bias': (64,),
            'fc2_weight': (output_dim, 64),  'fc2_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_lidar_output_dim(self):
        """Calculates the flattened dimension after LiDAR conv layers."""
        l = self.input_dim
        for i in range(1, 6):
            k = self.shapes[f'conv{i}_weight'][2]
            s = self.strides[f'conv{i}']
            l = (l - k) // s + 1
        c = self.shapes['conv5_weight'][0]
        return c * l

    def _initialize_weights(self):
        """Initializes weights."""
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                fan_out = shape[0] * (shape[2] if len(shape) > 2 else 1)
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def _encode_frame(self, scan, odom):
        """Encode a single frame."""
        import numpy as np
        
        x = relu(conv1d(scan, self.params['conv1_weight'], self.params['conv1_bias'], self.strides['conv1']))
        x = relu(conv1d(x, self.params['conv2_weight'], self.params['conv2_bias'], self.strides['conv2']))
        x = relu(conv1d(x, self.params['conv3_weight'], self.params['conv3_bias'], self.strides['conv3']))
        x = relu(conv1d(x, self.params['conv4_weight'], self.params['conv4_bias'], self.strides['conv4']))
        x = relu(conv1d(x, self.params['conv5_weight'], self.params['conv5_bias'], self.strides['conv5']))
        scan_feat = flatten(x)
        
        odom_feat = relu(linear(odom, self.params['state_fc_weight'], self.params['state_fc_bias']))
        
        combined = np.concatenate([scan_feat, odom_feat], axis=1)
        return relu(linear(combined, self.params['feature_proj_weight'], self.params['feature_proj_bias']))

    def _causal_conv1d(self, x, weight, bias, dilation):
        """Causal 1D convolution (pad left only)."""
        import numpy as np
        
        # x: (batch, channels, seq_len)
        # weight: (out_ch, in_ch, kernel_size)
        padding = (self.kernel_size - 1) * dilation
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, 0)), mode='constant')
        
        return conv1d_padded(x_padded, weight, bias, stride=1, padding=0)

    def __call__(self, scans, odoms):
        """Forward pass with sequence of frames.

        Args:
            scans (np.ndarray): Scan sequence of shape (batch, seq_len, scan_dim).
            odoms (np.ndarray): Odom sequence of shape (batch, seq_len, state_dim).

        Returns:
            np.ndarray: Output array of shape (batch, output_dim).
        """
        import numpy as np
        
        batch_size, seq_len, _ = scans.shape
        
        # Encode each frame
        features = []
        for t in range(seq_len):
            scan_t = scans[:, t:t+1, :]  # (batch, 1, scan_dim)
            odom_t = odoms[:, t, :]       # (batch, state_dim)
            feat_t = self._encode_frame(scan_t, odom_t)  # (batch, hidden_size)
            features.append(feat_t)
        
        # Stack: (batch, hidden_size, seq_len) for temporal conv
        x = np.stack(features, axis=2)
        
        # Apply TCN blocks with causal convolution
        for i in range(self.num_levels):
            dilation = 2 ** i
            residual = x
            
            # Two conv layers per block
            x = self._causal_conv1d(x, self.params[f'tcn{i}_conv1_weight'], 
                                    self.params[f'tcn{i}_conv1_bias'], dilation)
            x = relu(x)
            x = self._causal_conv1d(x, self.params[f'tcn{i}_conv2_weight'],
                                    self.params[f'tcn{i}_conv2_bias'], dilation)
            x = relu(x)
            
            # Residual connection
            x = x + residual
        
        # Take last timestep
        out = x[:, :, -1]  # (batch, hidden_size)
        
        # Output head
        out = relu(linear(out, self.params['fc1_weight'], self.params['fc1_bias']))
        return tanh(linear(out, self.params['fc2_weight'], self.params['fc2_bias']))


# =============================================================================
# Map-Enhanced Models (Image-Based Static Map)
# =============================================================================

class MapEncoderImage(nn.Module):
    """2D CNN encoder for static map image feature extraction.
    
    Processes a static map image (e.g., track boundaries) and outputs
    a fixed-size feature vector. Used by TinyLidarNetMapImage.
    
    Architecture:
    - 4 Conv2D layers with MaxPooling and BatchNorm
    - Global Average Pooling
    - FC layer to output_dim
    """
    
    def __init__(self, input_channels: int = 3, output_dim: int = 128):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Input: (B, 3, 224, 224)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)  # -> 112x112
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 56x56
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # -> 28x28
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 14x14
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # -> 14x14
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 7x7
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # -> 7x7
        self.bn4 = nn.BatchNorm2d(128)
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> 1x1
        self.fc = nn.Linear(128, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class TinyLidarNetMapImage(nn.Module):
    """Multi-modal CNN fusing LiDAR with static map image features.
    
    Uses Late Fusion approach:
    - LiDAR branch: Same as TinyLidarNet (5 Conv layers)
    - Map branch: 2D CNN encoder (MapEncoderImage) for static map image
    - Fusion: Concatenate features and pass through FC layers
    
    The map features are computed once and cached for efficient inference.
    """
    
    def __init__(
        self,
        input_dim: int = 1080,
        map_feature_dim: int = 128,
        output_dim: int = 2
    ):
        super().__init__()
        
        self.map_feature_dim = map_feature_dim
        
        # --- LiDAR Branch (same as TinyLidarNet) ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy)))))
            self.lidar_flatten_dim = out.view(1, -1).shape[1]
        
        # --- Map Branch ---
        self.map_encoder = MapEncoderImage(input_channels=3, output_dim=map_feature_dim)
        
        # Cached map features
        self.register_buffer('cached_map_features', None)
        
        # --- Fusion Head ---
        fusion_dim = self.lidar_flatten_dim + map_feature_dim
        
        self.fc1 = nn.Linear(fusion_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_map_image(self, map_image):
        """Precompute and cache map features."""
        self.map_encoder.eval()
        with torch.no_grad():
            self.cached_map_features = self.map_encoder(map_image)
    
    def forward(self, lidar, map_image=None):
        batch_size = lidar.size(0)
        
        # LiDAR branch
        x = F.relu(self.conv1(lidar))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        lidar_features = x.view(x.size(0), -1)
        
        # Map branch
        if map_image is not None:
            map_features = self.map_encoder(map_image)
        elif self.cached_map_features is not None:
            map_features = self.cached_map_features.expand(batch_size, -1)
        else:
            raise ValueError("No map image provided and no cached features available.")
        
        # Late fusion
        fused = torch.cat([lidar_features, map_features], dim=1)
        
        # Regression head
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class TinyLidarNetMapImageNp:
    """NumPy implementation of TinyLidarNetMapImage.
    
    Fuses LiDAR data with static map image features using pure NumPy inference.
    """
    
    def __init__(
        self,
        input_dim: int = 1080,
        map_feature_dim: int = 128,
        output_dim: int = 2
    ):
        self.input_dim = input_dim
        self.map_feature_dim = map_feature_dim
        self.output_dim = output_dim
        self.params = {}
        
        # Cached map features
        self.cached_map_features = None
        
        # LiDAR conv strides
        self.lidar_strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}
        
        # LiDAR branch shapes
        self.shapes = {
            'conv1_weight': (24, 1, 10),  'conv1_bias': (24,),
            'conv2_weight': (36, 24, 8),  'conv2_bias': (36,),
            'conv3_weight': (48, 36, 4),  'conv3_bias': (48,),
            'conv4_weight': (64, 48, 3),  'conv4_bias': (64,),
            'conv5_weight': (64, 64, 3),  'conv5_bias': (64,),
        }
        
        # Map encoder shapes (with BatchNorm)
        self.shapes.update({
            'map_encoder_conv1_weight': (32, 3, 7, 7), 'map_encoder_conv1_bias': (32,),
            'map_encoder_bn1_weight': (32,), 'map_encoder_bn1_bias': (32,),
            'map_encoder_bn1_running_mean': (32,), 'map_encoder_bn1_running_var': (32,),
            
            'map_encoder_conv2_weight': (64, 32, 5, 5), 'map_encoder_conv2_bias': (64,),
            'map_encoder_bn2_weight': (64,), 'map_encoder_bn2_bias': (64,),
            'map_encoder_bn2_running_mean': (64,), 'map_encoder_bn2_running_var': (64,),
            
            'map_encoder_conv3_weight': (128, 64, 3, 3), 'map_encoder_conv3_bias': (128,),
            'map_encoder_bn3_weight': (128,), 'map_encoder_bn3_bias': (128,),
            'map_encoder_bn3_running_mean': (128,), 'map_encoder_bn3_running_var': (128,),
            
            'map_encoder_conv4_weight': (128, 128, 3, 3), 'map_encoder_conv4_bias': (128,),
            'map_encoder_bn4_weight': (128,), 'map_encoder_bn4_bias': (128,),
            'map_encoder_bn4_running_mean': (128,), 'map_encoder_bn4_running_var': (128,),
            
            'map_encoder_fc_weight': (map_feature_dim, 128), 'map_encoder_fc_bias': (map_feature_dim,),
        })
        
        # Fusion head shapes
        lidar_flatten_dim = self._get_lidar_output_dim()
        fusion_dim = lidar_flatten_dim + map_feature_dim
        
        self.shapes.update({
            'fc1_weight': (100, fusion_dim), 'fc1_bias': (100,),
            'fc2_weight': (50, 100),         'fc2_bias': (50,),
            'fc3_weight': (10, 50),          'fc3_bias': (10,),
            'fc4_weight': (output_dim, 10),  'fc4_bias': (output_dim,),
        })
        
        self._initialize_weights()
    
    def _get_lidar_output_dim(self) -> int:
        l = self.input_dim
        kernels = [10, 8, 4, 3, 3]
        strides = [4, 4, 2, 1, 1]
        for k, s in zip(kernels, strides):
            l = (l - k) // s + 1
        return 64 * l
    
    def _initialize_weights(self):
        for name, shape in self.shapes.items():
            if name.endswith('_weight') and 'bn' not in name:
                if 'conv' in name:
                    if len(shape) == 4:
                        fan_out = shape[0] * shape[2] * shape[3]
                    else:
                        fan_out = shape[0] * shape[2]
                else:
                    fan_out = shape[0]
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias') and 'bn' not in name:
                self.params[name] = zeros_init(shape)
            elif 'bn' in name and name.endswith('_weight'):
                self.params[name] = ones_init(shape)
            elif 'bn' in name and name.endswith('_bias'):
                self.params[name] = zeros_init(shape)
            elif 'running_mean' in name:
                self.params[name] = zeros_init(shape)
            elif 'running_var' in name:
                self.params[name] = ones_init(shape)
    
    def encode_map(self, map_image):
        """Encode map image to feature vector.
        
        Args:
            map_image: Input array of shape (1, 3, 224, 224).
        
        Returns:
            Map features of shape (1, map_feature_dim).
        """
        import numpy as np
        
        # Conv1 + BN1 + ReLU + Pool1 (stride=2, pool=2 -> /4)
        x = conv2d_padded(map_image, self.params['map_encoder_conv1_weight'],
                          self.params['map_encoder_conv1_bias'], stride=(2, 2), padding=(3, 3))
        x = batch_norm2d(x, self.params['map_encoder_bn1_weight'], self.params['map_encoder_bn1_bias'],
                         self.params['map_encoder_bn1_running_mean'], self.params['map_encoder_bn1_running_var'])
        x = relu(x)
        x = max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))  # 224 -> 56
        
        # Conv2 + BN2 + ReLU + Pool2
        x = conv2d_padded(x, self.params['map_encoder_conv2_weight'],
                          self.params['map_encoder_conv2_bias'], stride=(2, 2), padding=(2, 2))
        x = batch_norm2d(x, self.params['map_encoder_bn2_weight'], self.params['map_encoder_bn2_bias'],
                         self.params['map_encoder_bn2_running_mean'], self.params['map_encoder_bn2_running_var'])
        x = relu(x)
        x = max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))  # 28 -> 14
        
        # Conv3 + BN3 + ReLU + Pool3
        x = conv2d_padded(x, self.params['map_encoder_conv3_weight'],
                          self.params['map_encoder_conv3_bias'], stride=(1, 1), padding=(1, 1))
        x = batch_norm2d(x, self.params['map_encoder_bn3_weight'], self.params['map_encoder_bn3_bias'],
                         self.params['map_encoder_bn3_running_mean'], self.params['map_encoder_bn3_running_var'])
        x = relu(x)
        x = max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))  # 14 -> 7
        
        # Conv4 + BN4 + ReLU
        x = conv2d_padded(x, self.params['map_encoder_conv4_weight'],
                          self.params['map_encoder_conv4_bias'], stride=(1, 1), padding=(1, 1))
        x = batch_norm2d(x, self.params['map_encoder_bn4_weight'], self.params['map_encoder_bn4_bias'],
                         self.params['map_encoder_bn4_running_mean'], self.params['map_encoder_bn4_running_var'])
        x = relu(x)
        
        # Global Average Pooling
        x = adaptive_avg_pool2d(x, output_size=(1, 1))
        x = flatten(x)
        
        # FC
        x = relu(linear(x, self.params['map_encoder_fc_weight'], self.params['map_encoder_fc_bias']))
        
        return x
    
    def set_map_image(self, map_image):
        """Precompute and cache map features."""
        self.cached_map_features = self.encode_map(map_image)
    
    def __call__(self, lidar, map_image=None):
        """Forward pass.
        
        Args:
            lidar: Input array of shape (batch_size, 1, input_dim).
            map_image: Optional map image of shape (batch_size, 3, 224, 224).
                       If None, uses cached features.
        
        Returns:
            Output array of shape (batch_size, output_dim).
        """
        import numpy as np
        
        batch_size = lidar.shape[0]
        
        # LiDAR branch
        x = relu(conv1d(lidar, self.params['conv1_weight'], self.params['conv1_bias'], self.lidar_strides['conv1']))
        x = relu(conv1d(x, self.params['conv2_weight'], self.params['conv2_bias'], self.lidar_strides['conv2']))
        x = relu(conv1d(x, self.params['conv3_weight'], self.params['conv3_bias'], self.lidar_strides['conv3']))
        x = relu(conv1d(x, self.params['conv4_weight'], self.params['conv4_bias'], self.lidar_strides['conv4']))
        x = relu(conv1d(x, self.params['conv5_weight'], self.params['conv5_bias'], self.lidar_strides['conv5']))
        lidar_features = flatten(x)
        
        # Map branch
        if map_image is not None:
            map_features = self.encode_map(map_image)
        elif self.cached_map_features is not None:
            map_features = np.broadcast_to(self.cached_map_features, (batch_size, self.map_feature_dim)).copy()
        else:
            raise ValueError("No map image provided and no cached features available.")
        
        # Late fusion
        fused = np.concatenate([lidar_features, map_features], axis=1)
        
        # Regression head
        x = relu(linear(fused, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        x = relu(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))
        
        return tanh(linear(x, self.params['fc4_weight'], self.params['fc4_bias']))


# =============================================================================
# Map-Enhanced Models (BEV Fusion)
# =============================================================================

class TinyLidarNetMap(nn.Module):
    """Multi-modal CNN model fusing LiDAR with BEV map and kinematic state.

    Architecture:
    - LiDAR branch: Conv1D encoder (same as TinyLidarNet)
    - BEV branch: Conv2D encoder for bird's eye view map
    - State branch: FC layer for kinematic state
    - Late fusion of all three branches
    """

    def __init__(
        self,
        input_dim: int = 1080,
        state_dim: int = 13,
        bev_size: int = 64,
        bev_channels: int = 2,
        output_dim: int = 2
    ):
        """Initializes TinyLidarNetMap.

        Args:
            input_dim: Size of the input LiDAR scan array.
            state_dim: Size of the kinematic state vector.
            bev_size: Size of the BEV grid (bev_size x bev_size).
            bev_channels: Number of BEV input channels.
            output_dim: Size of the output prediction.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.bev_size = bev_size
        self.bev_channels = bev_channels

        # --- LiDAR Branch (same as TinyLidarNet) ---
        self.lidar_conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.lidar_conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.lidar_conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.lidar_conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.lidar_conv5 = nn.Conv1d(64, 64, kernel_size=3)

        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, input_dim)
            x = self.lidar_conv5(self.lidar_conv4(self.lidar_conv3(
                self.lidar_conv2(self.lidar_conv1(dummy_lidar)))))
            lidar_flatten_dim = x.view(1, -1).shape[1]

        # --- BEV Branch (Conv2D encoder) ---
        self.bev_conv1 = nn.Conv2d(bev_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bev_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bev_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        with torch.no_grad():
            dummy_bev = torch.zeros(1, bev_channels, bev_size, bev_size)
            x = self.bev_conv3(self.bev_conv2(self.bev_conv1(dummy_bev)))
            bev_flatten_dim = x.view(1, -1).shape[1]
        
        self.bev_fc = nn.Linear(bev_flatten_dim, 256)

        # --- State Branch ---
        self.state_fc = nn.Linear(state_dim, 64)
        
        # --- Fusion Head ---
        # lidar_flatten_dim + 256 (bev) + 64 (state)
        fusion_dim = lidar_flatten_dim + 256 + 64
        
        self.fc1 = nn.Linear(fusion_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (He) initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, lidar, bev, state):
        """Forward pass with LiDAR, BEV map, and kinematic state inputs.

        Args:
            lidar: Input tensor of shape (batch_size, 1, input_dim).
            bev: Input tensor of shape (batch_size, bev_channels, bev_size, bev_size).
            state: Input tensor of shape (batch_size, state_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim) with Tanh activation.
        """
        # LiDAR branch
        x_lidar = F.relu(self.lidar_conv1(lidar))
        x_lidar = F.relu(self.lidar_conv2(x_lidar))
        x_lidar = F.relu(self.lidar_conv3(x_lidar))
        x_lidar = F.relu(self.lidar_conv4(x_lidar))
        x_lidar = F.relu(self.lidar_conv5(x_lidar))
        lidar_features = x_lidar.view(x_lidar.size(0), -1)
        
        # BEV branch
        x_bev = F.relu(self.bev_conv1(bev))
        x_bev = F.relu(self.bev_conv2(x_bev))
        x_bev = F.relu(self.bev_conv3(x_bev))
        x_bev = x_bev.view(x_bev.size(0), -1)
        bev_features = F.relu(self.bev_fc(x_bev))
        
        # State branch
        state_features = F.relu(self.state_fc(state))
        
        # Late fusion
        fused = torch.cat([lidar_features, bev_features, state_features], dim=1)
        
        # Regression head
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class TinyLidarNetMapNp:
    """NumPy implementation of TinyLidarNetMap.

    Multi-modal architecture that fuses LiDAR data with BEV map representation
    and kinematic state. Uses Late Fusion approach for pure NumPy inference.
    """

    def __init__(
        self,
        input_dim: int = 1080,
        state_dim: int = 13,
        bev_size: int = 64,
        bev_channels: int = 2,
        output_dim: int = 2
    ):
        """Initializes TinyLidarNetMapNp.

        Args:
            input_dim: Size of the input LiDAR scan array.
            state_dim: Size of the kinematic state vector.
            bev_size: Size of the BEV grid (bev_size x bev_size).
            bev_channels: Number of BEV input channels.
            output_dim: Size of the output prediction.
        """
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.bev_size = bev_size
        self.bev_channels = bev_channels
        self.output_dim = output_dim
        self.params = {}

        # Stride definitions for LiDAR conv layers
        self.lidar_strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}
        
        # BEV conv strides
        self.bev_strides = {'conv1': 2, 'conv2': 2, 'conv3': 2}

        # LiDAR branch shapes (same as TinyLidarNet)
        self.shapes = {
            'lidar_conv1_weight': (24, 1, 10),  'lidar_conv1_bias': (24,),
            'lidar_conv2_weight': (36, 24, 8),  'lidar_conv2_bias': (36,),
            'lidar_conv3_weight': (48, 36, 4),  'lidar_conv3_bias': (48,),
            'lidar_conv4_weight': (64, 48, 3),  'lidar_conv4_bias': (64,),
            'lidar_conv5_weight': (64, 64, 3),  'lidar_conv5_bias': (64,),
        }

        # BEV branch shapes (Conv2D with padding=1)
        self.shapes.update({
            'bev_conv1_weight': (16, bev_channels, 3, 3), 'bev_conv1_bias': (16,),
            'bev_conv2_weight': (32, 16, 3, 3),          'bev_conv2_bias': (32,),
            'bev_conv3_weight': (64, 32, 3, 3),          'bev_conv3_bias': (64,),
        })

        # Calculate flatten dimensions
        lidar_flatten_dim = self._get_lidar_output_dim()
        bev_flatten_dim = self._get_bev_output_dim()
        
        # BEV FC
        self.shapes.update({
            'bev_fc_weight': (256, bev_flatten_dim), 'bev_fc_bias': (256,),
        })

        # State branch
        self.shapes.update({
            'state_fc_weight': (64, state_dim), 'state_fc_bias': (64,),
        })

        # Fusion head
        fusion_dim = lidar_flatten_dim + 256 + 64
        
        self.shapes.update({
            'fc1_weight': (100, fusion_dim), 'fc1_bias': (100,),
            'fc2_weight': (50, 100),         'fc2_bias': (50,),
            'fc3_weight': (10, 50),          'fc3_bias': (10,),
            'fc4_weight': (output_dim, 10),  'fc4_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_lidar_output_dim(self) -> int:
        """Calculates the flattened dimension after LiDAR conv layers."""
        l = self.input_dim
        kernels = [10, 8, 4, 3, 3]
        strides = [4, 4, 2, 1, 1]
        channels = 64
        
        for k, s in zip(kernels, strides):
            l = (l - k) // s + 1
        
        return channels * l

    def _get_bev_output_dim(self) -> int:
        """Calculates the flattened dimension after BEV conv layers."""
        # Conv2D with padding=1, stride=2, kernel=3
        # output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
        # = floor((input_size + 2 - 3) / 2) + 1 = floor((input_size - 1) / 2) + 1
        h = w = self.bev_size
        for _ in range(3):
            h = (h + 2 * 1 - 3) // 2 + 1
            w = (w + 2 * 1 - 3) // 2 + 1
        
        return 64 * h * w

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (fan_out) and biases to zero."""
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                if 'conv' in name:
                    # Conv weight: use last dimension for fan_out calculation
                    if len(shape) == 4:  # Conv2D
                        fan_out = shape[0] * shape[2] * shape[3]
                    else:  # Conv1D
                        fan_out = shape[0] * shape[2]
                else:
                    fan_out = shape[0]
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def _conv2d_padded(self, x, weight, bias, stride, padding):
        """Apply 2D convolution with padding."""
        import numpy as np
        
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                      mode='constant', constant_values=0)
        
        return conv2d(x, weight, bias, stride=(stride, stride))

    def __call__(self, lidar, bev, state):
        """Performs the forward pass of the model.

        Args:
            lidar: Input array of shape (batch_size, 1, input_dim).
            bev: Input array of shape (batch_size, bev_channels, bev_size, bev_size).
            state: Input array of shape (batch_size, state_dim).

        Returns:
            Output array of shape (batch_size, output_dim).
        """
        import numpy as np
        
        # LiDAR branch
        x = relu(conv1d(lidar, self.params['lidar_conv1_weight'], 
                       self.params['lidar_conv1_bias'], self.lidar_strides['conv1']))
        x = relu(conv1d(x, self.params['lidar_conv2_weight'], 
                       self.params['lidar_conv2_bias'], self.lidar_strides['conv2']))
        x = relu(conv1d(x, self.params['lidar_conv3_weight'], 
                       self.params['lidar_conv3_bias'], self.lidar_strides['conv3']))
        x = relu(conv1d(x, self.params['lidar_conv4_weight'], 
                       self.params['lidar_conv4_bias'], self.lidar_strides['conv4']))
        x = relu(conv1d(x, self.params['lidar_conv5_weight'], 
                       self.params['lidar_conv5_bias'], self.lidar_strides['conv5']))
        lidar_features = flatten(x)
        
        # BEV branch
        x_bev = relu(self._conv2d_padded(bev, self.params['bev_conv1_weight'],
                                         self.params['bev_conv1_bias'], 
                                         stride=2, padding=1))
        x_bev = relu(self._conv2d_padded(x_bev, self.params['bev_conv2_weight'],
                                         self.params['bev_conv2_bias'],
                                         stride=2, padding=1))
        x_bev = relu(self._conv2d_padded(x_bev, self.params['bev_conv3_weight'],
                                         self.params['bev_conv3_bias'],
                                         stride=2, padding=1))
        x_bev = flatten(x_bev)
        bev_features = relu(linear(x_bev, self.params['bev_fc_weight'], 
                                   self.params['bev_fc_bias']))
        
        # State branch
        state_features = relu(linear(state, self.params['state_fc_weight'], 
                                     self.params['state_fc_bias']))
        
        # Late fusion (concatenate along feature dimension)
        fused = np.concatenate([lidar_features, bev_features, state_features], axis=1)
        
        # Regression head
        x = relu(linear(fused, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        x = relu(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))
        
        return tanh(linear(x, self.params['fc4_weight'], self.params['fc4_bias']))


# =============================================================================
# Ablation Study Models - Local, Global, and Dual BEV
# =============================================================================

class TinyLidarNetLocalBEV(nn.Module):
    """Pattern A: LiDAR + Local BEV + State fusion.
    
    Local BEV characteristics:
    - Vehicle-centered, rotated to vehicle heading
    - 64x64 grid, 2 channels (left/right boundaries)
    - Good for local perception and obstacle avoidance
    """

    def __init__(
        self,
        input_dim: int = 1080,
        state_dim: int = 13,
        local_bev_size: int = 64,
        local_bev_channels: int = 2,
        output_dim: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.local_bev_size = local_bev_size
        self.local_bev_channels = local_bev_channels

        # --- LiDAR Branch ---
        self.lidar_conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.lidar_conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.lidar_conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.lidar_conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.lidar_conv5 = nn.Conv1d(64, 64, kernel_size=3)

        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, input_dim)
            x = self.lidar_conv5(self.lidar_conv4(self.lidar_conv3(
                self.lidar_conv2(self.lidar_conv1(dummy_lidar)))))
            lidar_flatten_dim = x.view(1, -1).shape[1]

        # --- Local BEV Branch ---
        self.local_bev_conv1 = nn.Conv2d(local_bev_channels, 16, kernel_size=3, stride=2, padding=1)
        self.local_bev_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.local_bev_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        with torch.no_grad():
            dummy_bev = torch.zeros(1, local_bev_channels, local_bev_size, local_bev_size)
            x = self.local_bev_conv3(self.local_bev_conv2(self.local_bev_conv1(dummy_bev)))
            local_bev_flatten_dim = x.view(1, -1).shape[1]
        
        self.local_bev_fc = nn.Linear(local_bev_flatten_dim, 256)

        # --- State Branch ---
        self.state_fc = nn.Linear(state_dim, 64)
        
        # --- Fusion Head ---
        fusion_dim = lidar_flatten_dim + 256 + 64
        
        self.fc1 = nn.Linear(fusion_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, lidar, local_bev, state):
        # LiDAR branch
        x_lidar = F.relu(self.lidar_conv1(lidar))
        x_lidar = F.relu(self.lidar_conv2(x_lidar))
        x_lidar = F.relu(self.lidar_conv3(x_lidar))
        x_lidar = F.relu(self.lidar_conv4(x_lidar))
        x_lidar = F.relu(self.lidar_conv5(x_lidar))
        lidar_features = x_lidar.view(x_lidar.size(0), -1)
        
        # Local BEV branch
        x_bev = F.relu(self.local_bev_conv1(local_bev))
        x_bev = F.relu(self.local_bev_conv2(x_bev))
        x_bev = F.relu(self.local_bev_conv3(x_bev))
        x_bev = x_bev.view(x_bev.size(0), -1)
        local_bev_features = F.relu(self.local_bev_fc(x_bev))
        
        # State branch
        state_features = F.relu(self.state_fc(state))
        
        # Late fusion
        fused = torch.cat([lidar_features, local_bev_features, state_features], dim=1)
        
        # Regression head
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class TinyLidarNetGlobalBEV(nn.Module):
    """Pattern B: LiDAR + Global BEV + State fusion.
    
    Global BEV characteristics:
    - Map-fixed coordinates, no rotation
    - 128x128 grid, 3 channels (left/right boundaries + ego position)
    - Good for global planning and route understanding
    """

    def __init__(
        self,
        input_dim: int = 1080,
        state_dim: int = 13,
        global_bev_size: int = 128,
        global_bev_channels: int = 3,
        output_dim: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.global_bev_size = global_bev_size
        self.global_bev_channels = global_bev_channels

        # --- LiDAR Branch ---
        self.lidar_conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.lidar_conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.lidar_conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.lidar_conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.lidar_conv5 = nn.Conv1d(64, 64, kernel_size=3)

        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, input_dim)
            x = self.lidar_conv5(self.lidar_conv4(self.lidar_conv3(
                self.lidar_conv2(self.lidar_conv1(dummy_lidar)))))
            lidar_flatten_dim = x.view(1, -1).shape[1]

        # --- Global BEV Branch (larger network for bigger grid) ---
        self.global_bev_conv1 = nn.Conv2d(global_bev_channels, 16, kernel_size=3, stride=2, padding=1)
        self.global_bev_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.global_bev_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.global_bev_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        with torch.no_grad():
            dummy_bev = torch.zeros(1, global_bev_channels, global_bev_size, global_bev_size)
            x = self.global_bev_conv4(self.global_bev_conv3(
                self.global_bev_conv2(self.global_bev_conv1(dummy_bev))))
            global_bev_flatten_dim = x.view(1, -1).shape[1]
        
        self.global_bev_fc = nn.Linear(global_bev_flatten_dim, 256)

        # --- State Branch ---
        self.state_fc = nn.Linear(state_dim, 64)
        
        # --- Fusion Head ---
        fusion_dim = lidar_flatten_dim + 256 + 64
        
        self.fc1 = nn.Linear(fusion_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, lidar, global_bev, state):
        # LiDAR branch
        x_lidar = F.relu(self.lidar_conv1(lidar))
        x_lidar = F.relu(self.lidar_conv2(x_lidar))
        x_lidar = F.relu(self.lidar_conv3(x_lidar))
        x_lidar = F.relu(self.lidar_conv4(x_lidar))
        x_lidar = F.relu(self.lidar_conv5(x_lidar))
        lidar_features = x_lidar.view(x_lidar.size(0), -1)
        
        # Global BEV branch
        x_bev = F.relu(self.global_bev_conv1(global_bev))
        x_bev = F.relu(self.global_bev_conv2(x_bev))
        x_bev = F.relu(self.global_bev_conv3(x_bev))
        x_bev = F.relu(self.global_bev_conv4(x_bev))
        x_bev = x_bev.view(x_bev.size(0), -1)
        global_bev_features = F.relu(self.global_bev_fc(x_bev))
        
        # State branch
        state_features = F.relu(self.state_fc(state))
        
        # Late fusion
        fused = torch.cat([lidar_features, global_bev_features, state_features], dim=1)
        
        # Regression head
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class TinyLidarNetDualBEV(nn.Module):
    """Pattern C: LiDAR + Local BEV + Global BEV + State fusion.
    
    Dual BEV combines both local and global representations:
    - Local BEV: 64x64, 2 channels (vehicle-centered)
    - Global BEV: 128x128, 3 channels (map-fixed)
    - Best of both worlds for comprehensive understanding
    """

    def __init__(
        self,
        input_dim: int = 1080,
        state_dim: int = 13,
        local_bev_size: int = 64,
        local_bev_channels: int = 2,
        global_bev_size: int = 128,
        global_bev_channels: int = 3,
        output_dim: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim

        # --- LiDAR Branch ---
        self.lidar_conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.lidar_conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.lidar_conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.lidar_conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.lidar_conv5 = nn.Conv1d(64, 64, kernel_size=3)

        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, input_dim)
            x = self.lidar_conv5(self.lidar_conv4(self.lidar_conv3(
                self.lidar_conv2(self.lidar_conv1(dummy_lidar)))))
            lidar_flatten_dim = x.view(1, -1).shape[1]

        # --- Local BEV Branch ---
        self.local_bev_conv1 = nn.Conv2d(local_bev_channels, 16, kernel_size=3, stride=2, padding=1)
        self.local_bev_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.local_bev_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        with torch.no_grad():
            dummy_local = torch.zeros(1, local_bev_channels, local_bev_size, local_bev_size)
            x = self.local_bev_conv3(self.local_bev_conv2(self.local_bev_conv1(dummy_local)))
            local_bev_flatten_dim = x.view(1, -1).shape[1]
        
        self.local_bev_fc = nn.Linear(local_bev_flatten_dim, 256)

        # --- Global BEV Branch ---
        self.global_bev_conv1 = nn.Conv2d(global_bev_channels, 16, kernel_size=3, stride=2, padding=1)
        self.global_bev_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.global_bev_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.global_bev_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        with torch.no_grad():
            dummy_global = torch.zeros(1, global_bev_channels, global_bev_size, global_bev_size)
            x = self.global_bev_conv4(self.global_bev_conv3(
                self.global_bev_conv2(self.global_bev_conv1(dummy_global))))
            global_bev_flatten_dim = x.view(1, -1).shape[1]
        
        self.global_bev_fc = nn.Linear(global_bev_flatten_dim, 256)

        # --- State Branch ---
        self.state_fc = nn.Linear(state_dim, 64)
        
        # --- Fusion Head (larger for dual BEV) ---
        # lidar + local_bev + global_bev + state = 1792 + 256 + 256 + 64 = 2368
        fusion_dim = lidar_flatten_dim + 256 + 256 + 64
        
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(10, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, lidar, local_bev, global_bev, state):
        # LiDAR branch
        x_lidar = F.relu(self.lidar_conv1(lidar))
        x_lidar = F.relu(self.lidar_conv2(x_lidar))
        x_lidar = F.relu(self.lidar_conv3(x_lidar))
        x_lidar = F.relu(self.lidar_conv4(x_lidar))
        x_lidar = F.relu(self.lidar_conv5(x_lidar))
        lidar_features = x_lidar.view(x_lidar.size(0), -1)
        
        # Local BEV branch
        x_local = F.relu(self.local_bev_conv1(local_bev))
        x_local = F.relu(self.local_bev_conv2(x_local))
        x_local = F.relu(self.local_bev_conv3(x_local))
        x_local = x_local.view(x_local.size(0), -1)
        local_bev_features = F.relu(self.local_bev_fc(x_local))
        
        # Global BEV branch
        x_global = F.relu(self.global_bev_conv1(global_bev))
        x_global = F.relu(self.global_bev_conv2(x_global))
        x_global = F.relu(self.global_bev_conv3(x_global))
        x_global = F.relu(self.global_bev_conv4(x_global))
        x_global = x_global.view(x_global.size(0), -1)
        global_bev_features = F.relu(self.global_bev_fc(x_global))
        
        # State branch
        state_features = F.relu(self.state_fc(state))
        
        # Late fusion (all four branches)
        fused = torch.cat([lidar_features, local_bev_features, global_bev_features, state_features], dim=1)
        
        # Regression head
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


# =============================================================================
# NumPy Implementations for Ablation Study Models
# =============================================================================

class TinyLidarNetLocalBEVNp:
    """NumPy implementation of TinyLidarNetLocalBEV (Pattern A)."""

    def __init__(
        self,
        input_dim: int = 1080,
        state_dim: int = 13,
        local_bev_size: int = 64,
        local_bev_channels: int = 2,
        output_dim: int = 2
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.local_bev_size = local_bev_size
        self.local_bev_channels = local_bev_channels
        self.output_dim = output_dim
        self.params = {}

        # LiDAR branch
        self.lidar_strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}
        
        self.shapes = {
            'lidar_conv1_weight': (24, 1, 10),  'lidar_conv1_bias': (24,),
            'lidar_conv2_weight': (36, 24, 8),  'lidar_conv2_bias': (36,),
            'lidar_conv3_weight': (48, 36, 4),  'lidar_conv3_bias': (48,),
            'lidar_conv4_weight': (64, 48, 3),  'lidar_conv4_bias': (64,),
            'lidar_conv5_weight': (64, 64, 3),  'lidar_conv5_bias': (64,),
        }

        # Local BEV branch
        self.shapes.update({
            'local_bev_conv1_weight': (16, local_bev_channels, 3, 3), 'local_bev_conv1_bias': (16,),
            'local_bev_conv2_weight': (32, 16, 3, 3), 'local_bev_conv2_bias': (32,),
            'local_bev_conv3_weight': (64, 32, 3, 3), 'local_bev_conv3_bias': (64,),
        })

        lidar_flatten_dim = self._get_lidar_output_dim()
        local_bev_flatten_dim = self._get_local_bev_output_dim()
        
        self.shapes.update({
            'local_bev_fc_weight': (256, local_bev_flatten_dim), 'local_bev_fc_bias': (256,),
            'state_fc_weight': (64, state_dim), 'state_fc_bias': (64,),
        })

        fusion_dim = lidar_flatten_dim + 256 + 64
        self.shapes.update({
            'fc1_weight': (100, fusion_dim), 'fc1_bias': (100,),
            'fc2_weight': (50, 100), 'fc2_bias': (50,),
            'fc3_weight': (10, 50), 'fc3_bias': (10,),
            'fc4_weight': (output_dim, 10), 'fc4_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_lidar_output_dim(self) -> int:
        l = self.input_dim
        for k, s in zip([10, 8, 4, 3, 3], [4, 4, 2, 1, 1]):
            l = (l - k) // s + 1
        return 64 * l

    def _get_local_bev_output_dim(self) -> int:
        h = w = self.local_bev_size
        for _ in range(3):
            h = (h + 2 - 3) // 2 + 1
            w = (w + 2 - 3) // 2 + 1
        return 64 * h * w

    def _initialize_weights(self):
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                if 'conv' in name:
                    fan_out = shape[0] * (shape[2] * shape[3] if len(shape) == 4 else shape[2])
                else:
                    fan_out = shape[0]
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def _conv2d_padded(self, x, weight, bias, stride, padding):
        import numpy as np
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                      mode='constant', constant_values=0)
        return conv2d(x, weight, bias, stride=(stride, stride))

    def __call__(self, lidar, local_bev, state):
        import numpy as np
        
        # LiDAR branch
        x = relu(conv1d(lidar, self.params['lidar_conv1_weight'], 
                       self.params['lidar_conv1_bias'], self.lidar_strides['conv1']))
        x = relu(conv1d(x, self.params['lidar_conv2_weight'], 
                       self.params['lidar_conv2_bias'], self.lidar_strides['conv2']))
        x = relu(conv1d(x, self.params['lidar_conv3_weight'], 
                       self.params['lidar_conv3_bias'], self.lidar_strides['conv3']))
        x = relu(conv1d(x, self.params['lidar_conv4_weight'], 
                       self.params['lidar_conv4_bias'], self.lidar_strides['conv4']))
        x = relu(conv1d(x, self.params['lidar_conv5_weight'], 
                       self.params['lidar_conv5_bias'], self.lidar_strides['conv5']))
        lidar_features = flatten(x)
        
        # Local BEV branch
        x_bev = relu(self._conv2d_padded(local_bev, self.params['local_bev_conv1_weight'],
                                         self.params['local_bev_conv1_bias'], stride=2, padding=1))
        x_bev = relu(self._conv2d_padded(x_bev, self.params['local_bev_conv2_weight'],
                                         self.params['local_bev_conv2_bias'], stride=2, padding=1))
        x_bev = relu(self._conv2d_padded(x_bev, self.params['local_bev_conv3_weight'],
                                         self.params['local_bev_conv3_bias'], stride=2, padding=1))
        x_bev = flatten(x_bev)
        local_bev_features = relu(linear(x_bev, self.params['local_bev_fc_weight'], 
                                         self.params['local_bev_fc_bias']))
        
        # State branch
        state_features = relu(linear(state, self.params['state_fc_weight'], 
                                     self.params['state_fc_bias']))
        
        # Fusion
        fused = np.concatenate([lidar_features, local_bev_features, state_features], axis=1)
        
        x = relu(linear(fused, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        x = relu(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))
        return tanh(linear(x, self.params['fc4_weight'], self.params['fc4_bias']))


class TinyLidarNetGlobalBEVNp:
    """NumPy implementation of TinyLidarNetGlobalBEV (Pattern B)."""

    def __init__(
        self,
        input_dim: int = 1080,
        state_dim: int = 13,
        global_bev_size: int = 128,
        global_bev_channels: int = 3,
        output_dim: int = 2
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.global_bev_size = global_bev_size
        self.global_bev_channels = global_bev_channels
        self.output_dim = output_dim
        self.params = {}

        # LiDAR branch
        self.lidar_strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}
        
        self.shapes = {
            'lidar_conv1_weight': (24, 1, 10),  'lidar_conv1_bias': (24,),
            'lidar_conv2_weight': (36, 24, 8),  'lidar_conv2_bias': (36,),
            'lidar_conv3_weight': (48, 36, 4),  'lidar_conv3_bias': (48,),
            'lidar_conv4_weight': (64, 48, 3),  'lidar_conv4_bias': (64,),
            'lidar_conv5_weight': (64, 64, 3),  'lidar_conv5_bias': (64,),
        }

        # Global BEV branch (4 conv layers)
        self.shapes.update({
            'global_bev_conv1_weight': (16, global_bev_channels, 3, 3), 'global_bev_conv1_bias': (16,),
            'global_bev_conv2_weight': (32, 16, 3, 3), 'global_bev_conv2_bias': (32,),
            'global_bev_conv3_weight': (64, 32, 3, 3), 'global_bev_conv3_bias': (64,),
            'global_bev_conv4_weight': (64, 64, 3, 3), 'global_bev_conv4_bias': (64,),
        })

        lidar_flatten_dim = self._get_lidar_output_dim()
        global_bev_flatten_dim = self._get_global_bev_output_dim()
        
        self.shapes.update({
            'global_bev_fc_weight': (256, global_bev_flatten_dim), 'global_bev_fc_bias': (256,),
            'state_fc_weight': (64, state_dim), 'state_fc_bias': (64,),
        })

        fusion_dim = lidar_flatten_dim + 256 + 64
        self.shapes.update({
            'fc1_weight': (100, fusion_dim), 'fc1_bias': (100,),
            'fc2_weight': (50, 100), 'fc2_bias': (50,),
            'fc3_weight': (10, 50), 'fc3_bias': (10,),
            'fc4_weight': (output_dim, 10), 'fc4_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_lidar_output_dim(self) -> int:
        l = self.input_dim
        for k, s in zip([10, 8, 4, 3, 3], [4, 4, 2, 1, 1]):
            l = (l - k) // s + 1
        return 64 * l

    def _get_global_bev_output_dim(self) -> int:
        h = w = self.global_bev_size
        for _ in range(4):  # 4 conv layers
            h = (h + 2 - 3) // 2 + 1
            w = (w + 2 - 3) // 2 + 1
        return 64 * h * w

    def _initialize_weights(self):
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                if 'conv' in name:
                    fan_out = shape[0] * (shape[2] * shape[3] if len(shape) == 4 else shape[2])
                else:
                    fan_out = shape[0]
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def _conv2d_padded(self, x, weight, bias, stride, padding):
        import numpy as np
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                      mode='constant', constant_values=0)
        return conv2d(x, weight, bias, stride=(stride, stride))

    def __call__(self, lidar, global_bev, state):
        import numpy as np
        
        # LiDAR branch
        x = relu(conv1d(lidar, self.params['lidar_conv1_weight'], 
                       self.params['lidar_conv1_bias'], self.lidar_strides['conv1']))
        x = relu(conv1d(x, self.params['lidar_conv2_weight'], 
                       self.params['lidar_conv2_bias'], self.lidar_strides['conv2']))
        x = relu(conv1d(x, self.params['lidar_conv3_weight'], 
                       self.params['lidar_conv3_bias'], self.lidar_strides['conv3']))
        x = relu(conv1d(x, self.params['lidar_conv4_weight'], 
                       self.params['lidar_conv4_bias'], self.lidar_strides['conv4']))
        x = relu(conv1d(x, self.params['lidar_conv5_weight'], 
                       self.params['lidar_conv5_bias'], self.lidar_strides['conv5']))
        lidar_features = flatten(x)
        
        # Global BEV branch
        x_bev = relu(self._conv2d_padded(global_bev, self.params['global_bev_conv1_weight'],
                                         self.params['global_bev_conv1_bias'], stride=2, padding=1))
        x_bev = relu(self._conv2d_padded(x_bev, self.params['global_bev_conv2_weight'],
                                         self.params['global_bev_conv2_bias'], stride=2, padding=1))
        x_bev = relu(self._conv2d_padded(x_bev, self.params['global_bev_conv3_weight'],
                                         self.params['global_bev_conv3_bias'], stride=2, padding=1))
        x_bev = relu(self._conv2d_padded(x_bev, self.params['global_bev_conv4_weight'],
                                         self.params['global_bev_conv4_bias'], stride=2, padding=1))
        x_bev = flatten(x_bev)
        global_bev_features = relu(linear(x_bev, self.params['global_bev_fc_weight'], 
                                          self.params['global_bev_fc_bias']))
        
        # State branch
        state_features = relu(linear(state, self.params['state_fc_weight'], 
                                     self.params['state_fc_bias']))
        
        # Fusion
        fused = np.concatenate([lidar_features, global_bev_features, state_features], axis=1)
        
        x = relu(linear(fused, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        x = relu(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))
        return tanh(linear(x, self.params['fc4_weight'], self.params['fc4_bias']))


class TinyLidarNetDualBEVNp:
    """NumPy implementation of TinyLidarNetDualBEV (Pattern C)."""

    def __init__(
        self,
        input_dim: int = 1080,
        state_dim: int = 13,
        local_bev_size: int = 64,
        local_bev_channels: int = 2,
        global_bev_size: int = 128,
        global_bev_channels: int = 3,
        output_dim: int = 2
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.local_bev_size = local_bev_size
        self.local_bev_channels = local_bev_channels
        self.global_bev_size = global_bev_size
        self.global_bev_channels = global_bev_channels
        self.output_dim = output_dim
        self.params = {}

        # LiDAR branch
        self.lidar_strides = {'conv1': 4, 'conv2': 4, 'conv3': 2, 'conv4': 1, 'conv5': 1}
        
        self.shapes = {
            'lidar_conv1_weight': (24, 1, 10),  'lidar_conv1_bias': (24,),
            'lidar_conv2_weight': (36, 24, 8),  'lidar_conv2_bias': (36,),
            'lidar_conv3_weight': (48, 36, 4),  'lidar_conv3_bias': (48,),
            'lidar_conv4_weight': (64, 48, 3),  'lidar_conv4_bias': (64,),
            'lidar_conv5_weight': (64, 64, 3),  'lidar_conv5_bias': (64,),
        }

        # Local BEV branch
        self.shapes.update({
            'local_bev_conv1_weight': (16, local_bev_channels, 3, 3), 'local_bev_conv1_bias': (16,),
            'local_bev_conv2_weight': (32, 16, 3, 3), 'local_bev_conv2_bias': (32,),
            'local_bev_conv3_weight': (64, 32, 3, 3), 'local_bev_conv3_bias': (64,),
        })

        # Global BEV branch
        self.shapes.update({
            'global_bev_conv1_weight': (16, global_bev_channels, 3, 3), 'global_bev_conv1_bias': (16,),
            'global_bev_conv2_weight': (32, 16, 3, 3), 'global_bev_conv2_bias': (32,),
            'global_bev_conv3_weight': (64, 32, 3, 3), 'global_bev_conv3_bias': (64,),
            'global_bev_conv4_weight': (64, 64, 3, 3), 'global_bev_conv4_bias': (64,),
        })

        lidar_flatten_dim = self._get_lidar_output_dim()
        local_bev_flatten_dim = self._get_local_bev_output_dim()
        global_bev_flatten_dim = self._get_global_bev_output_dim()
        
        self.shapes.update({
            'local_bev_fc_weight': (256, local_bev_flatten_dim), 'local_bev_fc_bias': (256,),
            'global_bev_fc_weight': (256, global_bev_flatten_dim), 'global_bev_fc_bias': (256,),
            'state_fc_weight': (64, state_dim), 'state_fc_bias': (64,),
        })

        # Larger fusion head for dual BEV
        fusion_dim = lidar_flatten_dim + 256 + 256 + 64
        self.shapes.update({
            'fc1_weight': (256, fusion_dim), 'fc1_bias': (256,),
            'fc2_weight': (64, 256), 'fc2_bias': (64,),
            'fc3_weight': (10, 64), 'fc3_bias': (10,),
            'fc4_weight': (output_dim, 10), 'fc4_bias': (output_dim,),
        })

        self._initialize_weights()

    def _get_lidar_output_dim(self) -> int:
        l = self.input_dim
        for k, s in zip([10, 8, 4, 3, 3], [4, 4, 2, 1, 1]):
            l = (l - k) // s + 1
        return 64 * l

    def _get_local_bev_output_dim(self) -> int:
        h = w = self.local_bev_size
        for _ in range(3):
            h = (h + 2 - 3) // 2 + 1
            w = (w + 2 - 3) // 2 + 1
        return 64 * h * w

    def _get_global_bev_output_dim(self) -> int:
        h = w = self.global_bev_size
        for _ in range(4):
            h = (h + 2 - 3) // 2 + 1
            w = (w + 2 - 3) // 2 + 1
        return 64 * h * w

    def _initialize_weights(self):
        for name, shape in self.shapes.items():
            if name.endswith('_weight'):
                if 'conv' in name:
                    fan_out = shape[0] * (shape[2] * shape[3] if len(shape) == 4 else shape[2])
                else:
                    fan_out = shape[0]
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith('_bias'):
                self.params[name] = zeros_init(shape)

    def _conv2d_padded(self, x, weight, bias, stride, padding):
        import numpy as np
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                      mode='constant', constant_values=0)
        return conv2d(x, weight, bias, stride=(stride, stride))

    def __call__(self, lidar, local_bev, global_bev, state):
        import numpy as np
        
        # LiDAR branch
        x = relu(conv1d(lidar, self.params['lidar_conv1_weight'], 
                       self.params['lidar_conv1_bias'], self.lidar_strides['conv1']))
        x = relu(conv1d(x, self.params['lidar_conv2_weight'], 
                       self.params['lidar_conv2_bias'], self.lidar_strides['conv2']))
        x = relu(conv1d(x, self.params['lidar_conv3_weight'], 
                       self.params['lidar_conv3_bias'], self.lidar_strides['conv3']))
        x = relu(conv1d(x, self.params['lidar_conv4_weight'], 
                       self.params['lidar_conv4_bias'], self.lidar_strides['conv4']))
        x = relu(conv1d(x, self.params['lidar_conv5_weight'], 
                       self.params['lidar_conv5_bias'], self.lidar_strides['conv5']))
        lidar_features = flatten(x)
        
        # Local BEV branch
        x_local = relu(self._conv2d_padded(local_bev, self.params['local_bev_conv1_weight'],
                                           self.params['local_bev_conv1_bias'], stride=2, padding=1))
        x_local = relu(self._conv2d_padded(x_local, self.params['local_bev_conv2_weight'],
                                           self.params['local_bev_conv2_bias'], stride=2, padding=1))
        x_local = relu(self._conv2d_padded(x_local, self.params['local_bev_conv3_weight'],
                                           self.params['local_bev_conv3_bias'], stride=2, padding=1))
        x_local = flatten(x_local)
        local_bev_features = relu(linear(x_local, self.params['local_bev_fc_weight'], 
                                         self.params['local_bev_fc_bias']))
        
        # Global BEV branch
        x_global = relu(self._conv2d_padded(global_bev, self.params['global_bev_conv1_weight'],
                                            self.params['global_bev_conv1_bias'], stride=2, padding=1))
        x_global = relu(self._conv2d_padded(x_global, self.params['global_bev_conv2_weight'],
                                            self.params['global_bev_conv2_bias'], stride=2, padding=1))
        x_global = relu(self._conv2d_padded(x_global, self.params['global_bev_conv3_weight'],
                                            self.params['global_bev_conv3_bias'], stride=2, padding=1))
        x_global = relu(self._conv2d_padded(x_global, self.params['global_bev_conv4_weight'],
                                            self.params['global_bev_conv4_bias'], stride=2, padding=1))
        x_global = flatten(x_global)
        global_bev_features = relu(linear(x_global, self.params['global_bev_fc_weight'], 
                                          self.params['global_bev_fc_bias']))
        
        # State branch
        state_features = relu(linear(state, self.params['state_fc_weight'], 
                                     self.params['state_fc_bias']))
        
        # Fusion (all four branches)
        fused = np.concatenate([lidar_features, local_bev_features, global_bev_features, state_features], axis=1)
        
        x = relu(linear(fused, self.params['fc1_weight'], self.params['fc1_bias']))
        x = relu(linear(x, self.params['fc2_weight'], self.params['fc2_bias']))
        x = relu(linear(x, self.params['fc3_weight'], self.params['fc3_bias']))
        return tanh(linear(x, self.params['fc4_weight'], self.params['fc4_bias']))
