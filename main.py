from fastai.collab import *
from fastai.tabular.all import *
import pandas as pd
from model import DotProduct
from Trainer import Trainer
from DataLoader import CFDL

def main():
    path = untar_data(URLs.ML_100k)
    movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', usecols=(0,1), names=('movie', 'title'), header=None)
    ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names = ['user', 'movie', 'rating', 'timestamp'])
    ratings = ratings.merge(movies) 
   
    dls = CFDL(64, ratings)
    train, valid = dls.loader()
    n_users = len(ratings['user'].unique())
    n_movies = len(ratings['movie'].unique())
    n_factors = 5
    # Create create_paramss
    model = DotProduct(n_users, n_movies, n_factors)

    train = Trainer(model, train, valid, wd = 0.1, epoch=5)
    train.train_loop()
if __name__ == "__main__":
    main()