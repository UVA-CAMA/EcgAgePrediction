import torch.nn as nn
import torch.nn.functional as F

class SimpleResidualBlock(nn.Module):
    def __init__(self,downsample):
        super(SimpleResidualBlock,self).__init__()
        self.downsample = downsample
        self.stride = 2 if self.downsample else 1
        K = 9
        P = (K-1)//2
        self.conv1 = nn.Conv1d(in_channels=256,
                               out_channels=256,
                               kernel_size=K,
                               stride=self.stride,
                               padding=P,
                               bias=False)
        
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(in_channels=256,
                               out_channels=256,
                               kernel_size=K,
                               padding=P,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(256)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=2,stride=2)
            self.idfunc_1 = nn.Conv1d(in_channels=256,
                                      out_channels=256,
                                      kernel_size=1,
                                      bias=False)

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)

        x = x+identity
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self,nOUT):
        super(MultiHeadAttention,self).__init__()
        self.conv = nn.Conv1d(in_channels=12,
                              out_channels=256,
                              kernel_size=15,
                              padding=7,
                              stride=2,
                              bias=False)
        self.bn = nn.BatchNorm1d(256)
        self.activ_0 = nn.LeakyReLU()
        
        self.rb_0 = SimpleResidualBlock(downsample=True)
        self.rb_1 = SimpleResidualBlock(downsample=True)
        self.rb_2 = SimpleResidualBlock(downsample=True)
        self.rb_3 = SimpleResidualBlock(downsample=True)
        self.rb_4 = SimpleResidualBlock(downsample=True)

        self.mha = nn.MultiheadAttention(256,8)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc_1 = nn.Linear(256,nOUT)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ_0(x)

        x = self.rb_0(x)
        x = self.rb_1(x)
        x = self.rb_2(x)
        x = self.rb_3(x)
        x = self.rb_4(x)

        x = F.dropout(x,p=0.5,training=self.training)

        x = x.squeeze(2).permute(2,0,1)
        x,s = self.mha(x,x,x)
        x = x.permute(1,2,0)
        x = self.pool(x).squeeze(2)
        x = self.fc_1(x)
        
        return x