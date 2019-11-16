import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def assert_shape(tensor, shape):
    try:
        assert len(tensor.shape) == len(shape)

        for a, b in zip(tensor.shape, shape):
            if a == None or b == None:
                continue

            assert a == b

    except:
        raise AssertionError(
            "Tensor has shape {} but expected shape {}".format(tensor.shape, shape))


class LSUNGenerator(nn.Module):
    def __init__(self, input_size):
        super(LSUNGenerator, self).__init__()
        self.input_size = input_size
        
        self.fc1 = nn.Linear(input_size, 8*8*256, bias = False)
        self.bn1 = nn.BatchNorm1d(8*8*256)
        
        self.conv1 = nn.ConvTranspose2d(256, 128, 5, bias = False, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.ConvTranspose2d(128, 64, 6, stride=2, bias = False, padding=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2, bias = False, padding=2)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.ConvTranspose2d(32, 16, 6, stride=2, bias = True, padding=2)
        self.bn5 = nn.BatchNorm2d(16)
        
        self.conv5 = nn.ConvTranspose2d(16, 3, 3, stride=1, bias = True, padding=1)
        
    def forward(self, x):
        assert_shape(x, (None, self.input_size, 1, 1))

        x = x.reshape(-1, self.input_size)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 256, 8, 8)
        x = F.leaky_relu(self.bn2(self.conv1(x)))   
        x = F.leaky_relu(self.bn3(self.conv2(x)))    
        x = F.leaky_relu(self.bn4(self.conv3(x)))        
        x = F.leaky_relu(self.bn5(self.conv4(x)))        

        x = self.conv5(x)
        
        assert_shape(x, (None, 3, 64, 64))
        
        return x


class LSUNDiscriminator(nn.Module):
    def __init__(self):
        super(LSUNDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, stride=2, padding=2)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.dropout3 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(8*8*256, 1)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.conv1(x)))
        x = self.dropout2(F.leaky_relu(self.conv2(x)))
        x = self.dropout3(F.leaky_relu(self.conv3(x)))
        x = self.fc1(x.view(-1, 8*8*256))

        return x

class DCGANGenerator(nn.Module):
    def __init__(self, input_size):
        super(DCGANGenerator, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.ConvTranspose2d(input_size, 512, 4, stride=1, padding=0, bias = False)
        self.bn1 = nn.BatchNorm2d(512)
        
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, bias = False, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.ConvTranspose2d(128, 64, 4, stride=2, bias = False, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.ConvTranspose2d(64, 3, 4, stride=2, bias = True, padding=1)
        
    def forward(self, x):
        assert_shape(x, (None, self.input_size, 1, 1))
        
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))   
        x = F.leaky_relu(self.bn3(self.conv3(x)))  
        x = F.leaky_relu(self.bn4(self.conv4(x))) 
        x = self.conv5(x)       
        
        assert_shape(x, (None, 3, 64, 64))
        
        return x

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.dropout4 = nn.Dropout(p=0.3)

        self.conv5 = nn.Conv2d(512, 1, 4, stride=1, padding=0)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.conv1(x)))
        x = self.dropout2(F.leaky_relu(self.conv2(x)))
        x = self.dropout3(F.leaky_relu(self.conv3(x)))
        x = self.dropout4(F.leaky_relu(self.conv4(x)))
        x = self.conv5(x)

        return x

class MNISTGenerator(nn.Module):
    def __init__(self, input_size):
        super(MNISTGenerator, self).__init__()
        self.input_size = input_size
        
        self.fc1 = nn.Linear(input_size, 5*5*256, bias = False)
        self.bn1 = nn.BatchNorm1d(5*5*256)
        
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, bias = False, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, bias = False, padding=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv3 = nn.ConvTranspose2d(64, 1, 4, stride=2, bias = True, padding=1)
        
    def forward(self, x):
        assert_shape(x, (None, self.input_size, 1, 1))
        
        x = x.view(-1, self.input_size)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 256, 5, 5)
        x = F.leaky_relu(self.bn2(self.conv1(x)))        
        x = F.leaky_relu(self.bn3(self.conv2(x)))        
        x = self.conv3(x)
        
        assert_shape(x, (None, 1, 28, 28))
        
        return x

class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()        
        
        self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(7*7*128, 1)
        
    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.conv1(x)))
        x = self.dropout2(F.leaky_relu(self.conv2(x)))
        x = self.fc1(x.view(-1, 7*7*128))
        
        return x