import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv_query = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv_key = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv_value = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.conv_query(x)
        key = self.conv_key(x)
        value = self.conv_value(x)

        # Flatten spatial dimensions
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        # Compute attention scores
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention)

        # Apply attention to value
        attended_value = torch.bmm(value, attention.permute(0, 2, 1))

        # Reshape and return
        return attended_value.view(x.size())

class ConvDropoutNormNonlinAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDropoutNormNonlinAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.instnorm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.attention = AttentionBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.instnorm(x)
        x = self.lrelu(x)
        x = self.attention(x)
        return x

class StackedConvLayersAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackedConvLayersAttention, self).__init__()
        self.blocks = nn.Sequential(
            ConvDropoutNormNonlinAttention(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
