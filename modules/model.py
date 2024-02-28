import torch
from torch import nn
from modules.functions import func

def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))

class DotProduct(nn.Module):
    def __init__(self, n_users, n_items, n_factors, y_range=(0,5.5)):
        super().__init__()
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users,1])
        self.item_factors = create_params([n_items, n_factors])
        self.item_bias = create_params([n_items,1])
        self.y_range  = y_range
            
    def forward(self, x):
        users = self.user_factors[x[:,0]]
        items  = self.item_factors[x[:,1]]
        res = (users*items).sum(dim=1, keepdim=True)
        res += self.user_bias[x[:,0]] + self.item_bias[x[:,1]]
        return func.sigmoid_range(res, *self.y_range)
    

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1,output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        return x