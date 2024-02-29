from fastai.tabular.all import untar_data, URLs
import pandas as pd
from modules.model import DotProduct
from modules.Trainer import Trainer
from modules.DataLoader import CFDL
import pickle
import time
def init_data(ratings):
    users = sorted(list(ratings['user'].unique()))
    movies = sorted(list(ratings['movie'].unique()))
    users.insert(0, "#na#")
    movies.insert(0, "#na#")
    return users, movies


def main():
    path = untar_data(URLs.ML_100k)
    # merge on the end rather than before training. Movie names are useless to computer -__-
    #movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', usecols=(0,1), names=('movie', 'title'), header=None)
    ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names = ['user', 'movie', 'rating', 'timestamp']) 
    ratings = ratings.drop(['timestamp'], axis = 1)
    dls = CFDL(128, ratings)
    train, valid = dls.loader()
    users, movies= init_data(ratings)
    n_users = len(users)
    n_movies = len(movies)
    n_factors = 100
    model = DotProduct(n_users, n_movies, n_factors)#0.018
    train_model = Trainer(model, train, valid, lr=0.032, wd = 0.00001, epochs=601)
    trained_model = train_model.train_loop()
    with open(f"model{time.time()}.pkl", "wb") as file:
        pickle.dump(trained_model, file)
if __name__ == "__main__":
    main()