import torch
import torch.nn as nn
import numpy as np
import typing as tt

def torchify(arr: np.array) -> torch.FloatTensor:
    assert isinstance(arr, np.ndarray), 'Input is not a Numpy array'
    return torch.tensor(arr, dtype=torch.float32)

class CNN(torch.nn.Module):
    def __init__(self, input_shape: tt.Tuple[int, ...] = (84, 84, 4), output_shape: tt.Tuple[int, ...] = (1,), dropout=0.3):
        """
        Convolutional Neural Network for processing Image-like inputs.
        Args:
        - input_shape: (Height, Width, Channels)
        - output_shape: Tuple indicating the output shape (e.g., scalar value as (1,))
        - dropout: Dropout rate for regularization.
        """
        super(CNN, self).__init__()
        
        channels = input_shape[2]
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(in_features=64 * 7 * 7, out_features=512),  
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=output_shape[0])
        )
    
    def forward(self, torched: np.array) -> torch.FloatTensor:
        if torched.ndim == 3:
            torched = torched.unsqueeze(0)
        torched = torched.permute(0,3,1,2) #Reorder to (batch, channels, height, width)

        return self.model(torched)