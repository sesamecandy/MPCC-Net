import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init




class ChannelAttentionBranch(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=3):
        super(ChannelAttentionBranch, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels *2, in_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Global Average Pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)

        # Global Max Pooling
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)

        #avg_pool = avg_pool.unsqueeze(-1).permute(0, 2, 1)  # bs,1,c
        weight_avg = self.fc1(avg_pool)
        #max_pool = max_pool.unsqueeze(-1).permute(0, 2, 1)
        weight_max = self.fc1(max_pool)

        out = torch.cat([weight_avg, weight_max], dim=1)
        out = out.view(b, c*2, 1, 1)

        # Channel attention maps
        #avg_out = self.fc(avg_pool).view(b, c, 1, 1)
        #max_out = self.fc(max_pool).view(b, c, 1, 1)
        out = self.fc(out).view(b, c, 1, 1)

        # Combine and apply sigmoid
        #out = avg_out + max_out

        return out

class SpatialAttentionBranch(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBranch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # CAP (Cross-channel Average Pooling)
        x_avg = self.avg_pool(x)
        x_avg = x_avg.expand_as(x)

        # CMP (Cross-channel Max Pooling)
        x_max = self.max_pool(x)
        x_max = x_max.expand_as(x)

        # Concatenate X_cap and X_cmp along the channel dimension
        x_cat = torch.cat([x_avg, x_max], dim=1)

        # Process through convolutional layer and sigmoid function
        x_out = self.conv(x_cat)
        x_out = self.sigmoid(x_out)

        # Spatial attention map AS
        attention_map = x_out

        # Expand attention map to match the dimensions of X
        attention_map = attention_map.expand_as(x)

        # Element-wise multiplication with input X
        x_spatial = x * attention_map

        return x_spatial


class PCAM(nn.Module):
    def __init__(self, in_channels, num_splits=4):
        super(PCAM, self).__init__()
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.channel_attention_branches = nn.ModuleList(
            [ChannelAttentionBranch(in_channels) for _ in range(num_splits)]
        )

    def forward(self, x):
        b, c, h, w = x.size()
        split_height = h // self.num_splits

        # Split the input feature map along the height dimension
        splits = torch.split(x, split_height, dim=2)

        # Process each split with its respective channel attention branch
        new_features = []
        for i, split in enumerate(splits):
            ca_map = self.channel_attention_branches[i](split)
            ca_map = ca_map.expand_as(split)
            new_features.append(split * ca_map)

        # Concatenate the processed splits along the height dimension
        out = torch.cat(new_features, dim=2)

        return out

