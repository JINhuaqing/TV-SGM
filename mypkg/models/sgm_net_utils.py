# some useful blocks for neural network to approximate the SGM model
import torch.nn as nn
from torch.functional import F
from pprint import pprint

class SGMNetBase(nn.Module):
    def __init__(self):
        #! Must call the super() method to initialize the base class
        super().__init__()

    def initialize_weights(self):
        """
        Initialize weights of the model
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """get the number of parameters of the model"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        # get the number of parameters of each module
        module_params = {}
    
        def get_module_params(module, name=''):
            if list(module.children()):  # if the module has children
                for child_name, child in module.named_children():
                    full_name = f"{name}.{child_name}" if name else child_name
                    get_module_params(child, full_name)
            else:  # if the module has no children
                if name:  # if the module has a name
                    module_params[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
        get_module_params(self)
    
        return {
            'total': total_params,
            'modules': module_params
        }

class FCBlock(nn.Module):
    """FC Block"""
    def __init__(self, in_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        layers = [
            nn.Linear(in_dim, out_dim), 
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
            ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

        
class Conv1DResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.2):
        super().__init__()
        
        layers = [
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels), 
            nn.LeakyReLU(),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers += [
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        x = self.leaky_relu(x)
        
        return x