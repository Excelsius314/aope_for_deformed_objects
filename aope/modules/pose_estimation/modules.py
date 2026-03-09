
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class ShapeReconstruction(nn.Module):

    def __init__(self, n_points, feature_dim):
        super(ShapeReconstruction, self).__init__()

        # Regress per Part
        self.fn1 = nn.Linear(feature_dim, 1024)
        self.fn2 = nn.Linear(feature_dim, 512)
        self.fn3 = nn.Linear(feature_dim, 1024)
        self.fn_out = nn.Linear(1024, n_points * 3)

    def forward(self, x : Tensor):
        bs = x.shape[0]

        x = F.relu(self.fn1(x))
        x = F.relu(self.fn2(x))
        x = F.relu(self.fn3(x))

        return self.fn_out(x).view(bs, -1, 3)

class JointParameterPediction(nn.Module):

    def __init__(self, feature_dim):
        super(JointParameterPediction, self).__init__()
    
        # Regress from two adjacent parts
        self.fn1 = nn.Linear(2*feature_dim, feature_dim)
        self.fn2 = nn.Linear(feature_dim, 1024)
        self.fn3 = nn.Linear(feature_dim, 512)
        self.fn_out = nn.Linear(512, 6) # pivot point + Orientation

    def forward(self, x):
        x1 = F.relu(self.fn1(x))
        x2 = F.relu(self.fn2(x1))
        x3 = F.relu(self.fn3(x2))
        return self.fn_out(x3)
    


    

        

