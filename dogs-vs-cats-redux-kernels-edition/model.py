import os
import numpy as np
#from apex import amp

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50

class ResNet50Model():
    