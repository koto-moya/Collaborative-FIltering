from fastai.collab import *
from fastai.tabular.all import *
import pandas as pd

class CFDL():
    '''DataLoader for tabular data'''
    def __init__(self, bs, ds, valid_pct=0.2):
        self.bs = bs
        self.vs = valid_pct
        self.ds = ds
        self.v_idx = round(self.vs*len(self.ds))

    def shuffle(self, df):
        return df.sample(frac=1).reset_index(drop=True)
    
    def splitter(self):
        sdf = self.shuffle(self.ds)
        train = sdf[self.v_idx:]
        valid = sdf[:self.v_idx]
        return train.reset_index(drop=True), valid.reset_index(drop=True)
    
    def batcher(self, df):
        return [df[i*self.bs:self.bs*(i+1)] for i in range((len(df) + self.bs - 1)//self.bs)]
    
    def tuplizer(self, lst):
        return [(torch.tensor(df.iloc[:, 0:2].values), torch.tensor(df.iloc[:, 2:3].values)) for df in lst]

    def loader(self):
        train, valid = self.splitter()
        batched_train = self.tuplizer(self.batcher(train))
        batched_valid = self.tuplizer(self.batcher(valid))
        return batched_train, batched_valid 
