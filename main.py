from fastai.collab import *
from fastai.tabular.all import *
import pandas as pd


class DotProduct(Module):
    def __init__(self,n_users,  n_factors, n_items):
        self.user_factors = Embedding(n_users, n_factors)
        self.item_factors = Embedding(n_items, n_factors)

    def forward(self, x):
        users = self.user_factors(x[:,0])
        item = self.item_factors(x[:,0])
        return (users * item).sum(dim=1)

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
    
    def loader(self):
        train, valid = self.splitter()
        batched_train = self.batcher(train)
        batched_valid = self.batcher(valid)
        return batched_train, batched_valid 
    
    
class Trainer():
    def __init__(self, model, train_data, valid_data, lr = 0.003, epochs = 100):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.lr = lr
        self.epochs = epochs
        self.loss = torch.tensor([[0]])
    
    def train(self):
        for x,y in self.train_data:
            preds = self.model(x)
            self.loss = self.rmse(preds, y)
            self.loss.backward()
            self.update_params()

    def update_params(self):
        for param in self.model.parameters():
            param.data -= self.lr*param.grad.data
            param.grad = None

    def rmse(self, preds, y):
        return ((preds - y)**2).mean().sqrt()
def main():
    path = untar_data(URLs.ML_100k)
    movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', usecols=(0,1), names=('movie', 'title'), header=None)
    ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names = ['user', 'movie', 'rating', 'timestamp'])
    ratings = ratings.merge(movies) 
    #dls = CollabDataLoaders.from_df(ratings=ratings,user_name="user", item_name="title",rating_name="rating", bs=64)
    dls = CFDL(64, ratings)
    train, valid = dls.loader()
    print(train[10])
    #n_users = len(dls.classes['user'])
    #n_movies = len(dls.classes['title'])
    #n_factors = 5
    # x is [[user_n,movie_n],] y is [[rating_n],] where N is bs

    # Create embeddings
    #model = DotProduct(n_users, n_movies, n_factors)
if __name__ == "__main__":
    main()