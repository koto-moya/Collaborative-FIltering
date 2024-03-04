from fastai.tabular.all import untar_data, URLs
import pandas as pd
from modules.model import DotProduct, CollabNN
from modules.Trainer import Trainer
from modules.DataLoader import CFDL
from modules.functions import func
import pickle
import time
import re
import torch
def init_data(ratings, user, item):
    users = sorted(list(ratings[user].unique()))
    movies = sorted(list(ratings[item].unique()))
    users.insert(0, "#na#")
    movies.insert(0, "#na#")
    return users, movies

def nn_model():
    path = untar_data(URLs.ML_100k)
    # merge on the end rather than before training. Movie names are useless to computer -__-
    #movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', usecols=(0,1), names=('movie', 'title'), header=None)
    ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names = ['user', 'movie', 'rating', 'timestamp']) 
    ratings = ratings[['user', 'movie', 'rating']] #'timestamp'
    dls = CFDL(ratings)
    emdsz = dls.embed_size()
    train, valid = dls.loader()
    model = CollabNN(emdsz)
    train_model = Trainer(model, train, valid, lr=0.003, epochs=100, loss_func=func.cross_entropy_loss)
    trained_model = train_model.train_loop()

def dot_model():
    path = untar_data(URLs.ML_100k)
    ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names = ['user', 'movie', 'rating', 'timestamp']) 
    ratings = ratings[['user', 'movie', 'rating']] #'timestamp'
    dls = CFDL(ratings)
    train, valid = dls.loader()
    users, movies= init_data(ratings)
    n_users = len(users)
    n_movies = len(movies)
    n_factors = 100
    model = DotProduct(n_users, n_movies, n_factors)#0.018
    train_model = Trainer(model, train, valid, model_type='dot_prod', lr=0.018, wd = 0.00001, epochs=501, loss_func=func.mse)
    trained_model = train_model.train_loop()
    #with open(f"model{round(time.time())}.pkl", "wb") as file:
      #  pickle.dump(trained_model, file)
    
def rdat(name):
    with open(f'ml-1m/{name}.dat', 'r', encoding='iso-8859-1') as file:
        data = file.read()
    data = re.split(r'\n', data)
    data =  [re.split(r'::', row) for row in data]
    return data

def one_mil():
    names = ['movies', 'ratings', 'users']
    all_data = [rdat(name) for name in names]
    data = [pd.DataFrame(data = tables[1:], columns=tables[0]) for tables in all_data]
    ratings = data[1][['UserID', 'MovieID', 'Rating']]
    movies = data[0]
    ratings = ratings[ratings['UserID'] != '']
    ratings = ratings[ratings['MovieID'] != '']
    ratings = ratings.astype({'UserID': 'int', 'MovieID': 'int', 'Rating': 'int'})


    dls = CFDL(ratings)
    train, valid = dls.loader()
    users, movies= init_data(ratings, "UserID", "MovieID")
    n_users = max(ratings['UserID'].unique())+1 
    n_movies = max(ratings['MovieID'].unique())+1 
    n_factors = 100
    model = DotProduct(n_users, n_movies, n_factors)#0.018
    train_model = Trainer(model, train, valid, model_type='dot_prod', lr=0.061, wd = 0.000001, epochs=101, loss_func=func.rmse)
    trained_model = train_model.train_loop()


if __name__ == "__main__":
    one_mil()