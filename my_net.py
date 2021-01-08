import torch
import torch.nn as nn

def activation(name):
    '''
    Activation function switcher
    '''
    if name.upper() == 'TANH':
        return nn.Tanh()
    elif name.upper() == 'RELU':
        return nn.ReLU(inplace=True)
    elif name in ['leaky_relu', 'LeakyReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name.upper() == 'SIGMOID':
        return nn.Sigmoid()
    elif name.upper() == 'SOFTPLUS':
        return nn.Softplus()
    else: 
        raise ValueError(f'Unknown activation function: {name}')

class ConvBlock(nn.Sequential):
    '''
    Convolutional Block
    '''
    def __init__(self, in_channels, out_channels, pool=True, act_name = 'relu',drop_rate=0.):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True))
        self.add_module('act', activation(act_name))
        self.add_module('norm', nn.GroupNorm(4,out_channels))
        if pool:
            self.add_module('pool', nn.MaxPool2d(kernel_size=2))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout(p=drop_rate))

class FcBlock(nn.Sequential):
    '''
    Fully Connected Block
    '''
    def __init__(self, dim_in, dim_out, act_name = 'relu', drop_rate=0.):
        super().__init__()
        self.add_module('fc', nn.Linear(dim_in, dim_out, bias = True))
        self.add_module('act', activation(act_name))
        self.add_module('norm', nn.GroupNorm(4,dim_out))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout(p=drop_rate))

class Net(nn.Module):
    def __init__(self, act_name = 'relu', init_name = 'kaiming_normal', drop_rate = 0.):
        super(Net, self).__init__()
        model = nn.Sequential()
        model.add_module('conv_block1', ConvBlock(3,  32, pool=False))
        model.add_module('conv_block2', ConvBlock(32, 32, pool=True,  drop_rate=drop_rate))
        model.add_module('conv_block3', ConvBlock(32, 64, pool=False))
        model.add_module('conv_block4', ConvBlock(64, 64, pool=True, drop_rate=drop_rate))
        model.add_module('flatten',nn.Flatten(start_dim=1, end_dim=-1))
        model.add_module('fc_block1', FcBlock(64*32*32, 1024, drop_rate=drop_rate))
        model.add_module('fc_block2',FcBlock(1024, 256 ,drop_rate = drop_rate))
        model.add_module('last_layer', nn.Linear(256, 10, bias = True))        
        self.model = model
        
        if init_name is not None:
            self.init_weight(init_name)
            
    def forward(self, x):
        return self.model(x)
            
    def init_weight(self, name):
        if name == 'xavier_normal':
            nn_init = nn.init.xavier_normal_
        elif name == 'xavier_uniform':
            nn_init = nn.init.xavier_uniform_
        elif name == 'kaiming_normal':
            nn_init = nn.init.kaiming_normal_
        elif name == 'kaiming_uniform':
            nn_init =  nn.init.kaiming_uniform_
        else:
            raise ValueError(f'unknown initialization function: {name}')
    
        for param in self.parameters():
            if len(param.shape) > 1:
                nn_init(param)
                
    def model_size(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params 
