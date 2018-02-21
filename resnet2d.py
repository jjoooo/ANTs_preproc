import torch
import torch.nn as nn
import torchvision.models as models

resnet = models.resnet50(pretrained=False)

class Resnet(nn.Module):
    def __init__(self, num_mode):
        super(Resnet,self).__init__()
        
        self.layer0 = nn.Conv2d(num_mode,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layer1 = nn.Sequential(*list(resnet.children())[1:-2])
        
    def forward(self,x):
        out = self.layer0(x)
        out = self.layer1(out)

        return out

class Classifier(nn.Module):
    def __init__(self, batch_size):
        super(Classifier,self).__init__()
        self.batch_size = batch_size
        self.layer0 = nn.Sequential(
            nn.Conv2d(4096,100,kernel_size=1,bias=False),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,1,kernel_size=1,bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        out = self.layer0(x)
        out = out.view(self.batch_size,1)
        return out

class Segmentation(nn.Module):
    def __init__(self, batch_size):
        super(Segmentation,self).__init__()
        self.batch_size = batch_size
        self.layer0 = nn.Sequential(
            nn.Conv2d(4096,100,kernel_size=1,bias=False),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,1,kernel_size=1,bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        out = self.layer0(x)
        out = out.view(self.batch_size,1)
        return out