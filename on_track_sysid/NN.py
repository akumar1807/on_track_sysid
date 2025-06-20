import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, weight_decay=0.01):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(4,8) #Input Layer
        self.hl1 = nn.Linear(8, 8) # First hidden layer
        self.l2 = nn.Linear(8, 2) # Output layer
        self.relu = nn.LeakyReLU()

        self.initialize_weights()

        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.l1(x) #x is now a tensor
        x = self.relu(x) #Apply activation to tensor x
        x = self.hl1(x) #Pass to hidden layer
        x = self.relu(x) #Apply activation
        x = self.l2(x) 
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

