from fastai.collab import *
from fastai.tabular.all import *
import pandas as pd
from model import DotProduct
from Trainer import Trainer
from DataLoader import CFDL

def main():
    path = untar_data(URLs.ML_100k)
    
    # merge on the end rather than before training. Movie names are useless to computer -__-
    #movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', usecols=(0,1), names=('movie', 'title'), header=None)

    ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names = ['user', 'movie', 'rating', 'timestamp']) 
    ratings = ratings.drop(['timestamp'], axis = 1)
    dls = CFDL(64, ratings)
    train, valid = dls.loader()
    n_users = len(ratings['user'].unique())
    n_movies = len(ratings['movie'].unique())
    n_factors = 50
    # Create create_paramss
    model = DotProduct(n_users, n_movies, n_factors)
    train_model = Trainer(model, train, valid, wd = 0.1, epochs=30)
    train_model.train_loop()
if __name__ == "__main__":
    main()