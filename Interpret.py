import pickle
import pandas as pd
from fastai.tabular.all import untar_data, URLs
import numpy as np
import torch
from modules.DataLoader import CFDL


def main():
    path = untar_data(URLs.ML_100k)
    # merge on the end rather than before training. Movie names are useless to computer -__-
    movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', usecols=(0,1), names=('movie', 'title'), header=None)
    ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names = ['user', 'movie', 'rating', 'timestamp']) 
    ratings = ratings.drop(['timestamp'], axis = 1)
    table = ratings.merge(movies)
    mean_ratings = table[["title", "rating"]].groupby(by="title").mean()

    loader = CFDL()

    #print(mean_ratings.sort_values(by='rating', ascending=False))
    # Dataset Stats
    #mean_rating = torch.mean(rating_values, dtype=torch.float32)
    #variance = torch.sum(((rating_values - mean_rating)**2))/len(rating_values)
    #std_dev = np.sqrt(variance)
    #print(f"Dataset Stats: \nMean: {mean_rating} | variance: {variance}\nStandard deviation: {std_dev}") 

    with open("pickles/model3.pkl", "rb") as file:
        model = pickle.load(file)
    print("Model layers: ", model)
    print("Model Type:", type(model))
    print("Y Range:", model.y_range)
    for name, param in model.named_parameters():
        print(name, param.size(), param.device)
        #item_bias = model.item_bias.squeeze()
    #idxs = item_bias.argsort()[-5:]
    #for i in idxs:
    #   low_bias = table[table['movie'] == i.item()]
    #   print(low_bias['title'].unique())
if __name__ =="__main__":
    main() 