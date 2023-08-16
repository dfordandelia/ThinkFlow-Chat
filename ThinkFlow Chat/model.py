from numpy import dtype, float32
import torch
import torch.nn as nn

class myNeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size , num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)#,dtype=torch.float32)
        self.layer2 = nn.Linear(hidden_size,hidden_size)#,dtype=torch.float32)
        self.layer3 = nn.Linear(hidden_size,hidden_size)#,dtype=torch.float32)
        self.layer4 = nn.Linear(hidden_size,num_classes)#,dtype=torch.float32)
        self.acti1 = nn.ReLU()
        self.acti2 = nn.Softmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.acti1(out) 
        out = self.layer2(out)
        out = self.acti1(out)
        out = self.layer3(out)
        out = self.acti1(out)
        out = self.layer4(out)
        
        return out


        
