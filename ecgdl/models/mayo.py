import torch
from torch import nn

class MayoModelFCBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.8)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x
    
class MayoModelConvBlock(nn.Module):
    def __init__(self, channels_in, samples_in, filter_count, kernel_size, maxpool_factor):
        super().__init__()
        
        self.conv = nn.Conv1d(channels_in, filter_count, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(filter_count)
        self.activation = nn.ReLU()
        self.mp = nn.MaxPool1d(maxpool_factor)
        self.channels_out = filter_count
        self.samples_out = samples_in // maxpool_factor
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.mp(x)
        return x

class MayoModel(nn.Module):
    def __init__(self, input_channels, input_samples, pad_to, output_dim):
        super().__init__()
        
        self.padding = nn.ZeroPad1d((0, pad_to - input_samples))
        
        samples_in = pad_to
        channels_in = input_channels
        
        settings = [
            (7, 16, 2),
            (5, 16, 4),
            (5, 32, 2),
            (5, 32, 4),
            (5, 64, 2),
            (3, 64, 2),
            (3, 64, 2),
            (3, 64, 2),
        ]
        
        self.temporal_blocks = nn.ModuleList()
        for (k,n,mp) in settings:
            b = MayoModelConvBlock(channels_in, samples_in, n, k, mp)
            self.temporal_blocks.append(b)
            channels_in = b.channels_out
            samples_in = b.samples_out
        
        # spatial layer
        
        self.spatial_block = MayoModelConvBlock(samples_in, channels_in, 128, 12, 2)
        
        samples_in = self.spatial_block.samples_out
        channels_in = self.spatial_block.channels_out
        
        #final blocks
        current_dimension = samples_in*channels_in
        self.flatten = nn.Flatten()
        self.fc_blocks = nn.ModuleList()
        for n in [128, 64]:
            self.fc_blocks.append(MayoModelFCBlock(current_dimension, n))
            current_dimension = n
        self.head = nn.Linear(current_dimension, output_dim)

    def forward(self, x):
        x = self.padding(x)
        
        for b in self.temporal_blocks:
            x = b(x)
            
        x = x.permute(0, 2, 1)
        
        x = self.spatial_block(x)        
        x = self.flatten(x)
                
        for b in self.fc_blocks:
            x = b(x)
        x = self.head(x)
        return x