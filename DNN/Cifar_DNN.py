import torch.nn as nn
import torch.nn.functional as F

class CifarDNN(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(CifarDNN,self).__init__()
        self.hidden1 = nn.Linear(inputSize, 2000)
        self.hidden2 = nn.Linear(2000, 600)
        self.hidden3 = nn.Linear(600, 200)
        self.hidden4 = nn.Linear(200, 50)
        self.output = nn.Linear(50, outputSize)

    def forward(self, x):
        x = x.reshape(-1, 3072)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        x = F.relu(x)
        x = self.output(x)
        return x
