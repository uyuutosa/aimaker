import torch

class ResizeConv(torch.nn.Module):
    def __init__(self, n_in, n_out, scale_factor=2):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=0)
        self.norm = torch.nn.BatchNorm2d(n_out)
        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DownSample(torch.nn.Module):
    def __init__(self, n_in, n_out, scale_factor=2):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=scale_factor, padding=0)
        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class ConvBlock(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=0)
        self.norm = torch.nn.BatchNorm2d(n_out)
        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.pad(x) 
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x 
    
class ResBlock(torch.nn.Module):
    def __init__(self, n_in, is_skip=True):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(n_in, n_in, kernel_size=3, stride=1, padding=0)
        self.norm = torch.nn.BatchNorm2d(n_in)
        self.act = torch.nn.ReLU()
        self.is_skip = is_skip
    
    def forward(self, x):
        x_ini = x
        x = self.pad(x) 
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.is_skip:
            return x + x_ini
        else:
            return x
