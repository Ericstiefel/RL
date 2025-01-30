
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectFourNet(nn.Module):
    def __init__(self, input_channels=1, board_size=(6, 7), num_filters=128):
        super(ConnectFourNet, self).__init__()
        
        self.board_height, self.board_width = board_size
        self.num_actions = self.board_width  # Actions correspond to columns

        # Shared Convolutional Feature Extractor
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        
        #Normalize the output
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.bn4 = nn.BatchNorm2d(num_filters)

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)  # Reduce channels to 2
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_height * self.board_width, self.num_actions)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)  # Reduce channels to 1
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_height * self.board_width, 64)
        self.value_fc2 = nn.Linear(64, 1)  # Output single value

    def forward(self, x):
        # Shared Feature Extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  
        p = self.policy_fc(p)
        p = F.softmax(p, dim=1)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) 
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # Output between -1 and 1

        return p, v  