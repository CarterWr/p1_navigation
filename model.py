#Imports for the model
import numpy as np
import torch#import stuff such as functional later when needed just doing the base imports
import random
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Agent Model"""
    
    def __init__(self, state_size, action_size, seed, fc_1=64, fc_2=64, dropout_p=0.5):
        """
        class initalization
        
        Params
        - state_size(int): The amount of states in a given enviroment
        - action_size(int): The number of actions a agent can take
        - seed(int): random seed(random number)
        - fc_1(int): The number of nodes in the first hidden layer
        - fc_2(int): THe number of nodes in the second hidden layer
        - dropout_p(float): the percentage that a random neruon gets dropped out
               
        """
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_1)
        self.fc2 = nn.Linear(fc_1, fc_2)
        self.fc3 = nn.Linear(fc_2, action_size)
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, state):
        """
        The Forward pass that maps state to action values including dropout for regulization
        """
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)