import torch
from torch import nn
from modules.functions import func

def create_params(size): #"cpu")
    device = torch.device("cpu")#"mps" if torch.backends.mps.is_available() else "cpu")
    return nn.Parameter(torch.zeros(*size, device=device).normal_(0, 0.01))

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
    

class CollabNN(nn.Module):
    def __init__(self, embds ,n_act = 300, y_range=(0,5.5)):
        super().__init__()
        self.user_factors = create_params([*embds['user']])
        # item factors is a list of differntly sized tensors, 1 for each 'item'. tensor shape (len(set(item)), # of factors)
        self.item_factors = [create_params([*value]) for key,value in embds.items() if key != 'user']
        self.item_sz = sum([value[1] for key,value in embds.items() if key != 'user'])
        self.layers = nn.Sequential(
            nn.Linear(embds['user'][1]+self.item_sz, n_act),
            nn.ReLU(),
            nn.Linear(n_act, 5)
        )
        self.y_range  = y_range

    def forward(self, x):
        #print(x.shape, len(self.item_factors[0]))
        embds = self.user_factors[x[:,0]], torch.cat([self.item_factors[i-1][x[:,i]] for i in range(x.shape[1]) if i != 0], dim=1)
        #print(torch.cat(embds, dim=1).shape)
        x = self.layers(torch.cat(embds, dim=1))
        return func.sigmoid_range(x, *self.y_range)