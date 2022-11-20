import torch.nn as nn
from torchvision import models

def my_net():
    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier[6] = nn.Linear(4096, 3)
    return vgg16