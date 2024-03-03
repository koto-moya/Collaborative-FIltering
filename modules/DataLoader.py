import torch

class CFDL():
    '''DataLoader for tabular data'''
    def __init__(self, ds, nn=False, bs=64, valid_pct=0.2):
        self.bs = bs
        self.ds = ds
        self.v_idx = round(valid_pct*len(ds))
        self.nn = nn

    def shuffle(self, df):
        return df.sample(frac=1).reset_index(drop=True)
    
    def splitter(self):
        sdf = self.shuffle(self.ds)
        train = sdf[self.v_idx:]
        valid = sdf[:self.v_idx]
        return train.reset_index(drop=True), valid.reset_index(drop=True)
    
    def batcher(self, df):
        return [df[i*self.bs:self.bs*(i+1)] for i in range((len(df) + self.bs - 1)//self.bs)]
    
    
    def tuplizer(self, lst): # "cpu")
        # Need User Column, item columns, prediction column
        device = torch.device("cpu")#"mps" if torch.backends.mps.is_available() else "cpu")
        return [(torch.tensor(df.iloc[:, 0:2].values, device=device), torch.tensor(df.iloc[:, 2:3].values, device=device)) for df in lst]

    def loader(self):
        train, valid = self.splitter()
        batched_train = self.tuplizer(self.batcher(train))
        batched_valid = self.tuplizer(self.batcher(valid)) 
        return batched_train, batched_valid 
    
    def embed_size(self, log=False):
        '''Prediction column should always be the last column'''
        pred_col = ['rating']
        if log:
            return {f"{self.ds.columns[i]}":
                    (len(set(self.ds.iloc[:,i]))+1, 
                    int(torch.ceil(1.6*torch.log(torch.tensor(len(set(self.ds.iloc[:,i]))+1))).item())) 
                    for i in range(len(self.ds.columns)) if self.ds.columns[i] not in pred_col
                    } 
        else:
            return {f"{self.ds.columns[i]}":
                    (len(set(self.ds.iloc[:,i]))+1, 
                     min(600,round(1.6*(len(set(self.ds.iloc[:,i]))+1)**0.56))) 
                     for i in range(len(self.ds.columns)) if self.ds.columns[i] not in pred_col
                     }


