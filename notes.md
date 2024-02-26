# Dot Products

Taking the dot product of two vectors gives you a magnitude that we can think of as analogous to the cosine similarity of the vectors. 

In collaborative filtering, the values of the vectors are known as *latent factors*

**Latent Factors** - These are the underlying features of your data.  You can assign an arbitrary number of these to your dataset. 

each component of the data has an arbitrary array associated with it 

given a table
```
table = {'UIDs' : [id1,id2,id3,id4,id5], 
        'MovieIDs': [mid5,mid7,mid3,mid9,mid3], 
        'Movie score': [1.0, 2.5, 3., 5, 4.5]}
```
Each MovieID would have an array of latent factors, each UID would also have an associated latent factor.

THIS IS WRONG: I think this is getting at principle component analysis.  At least we are working with the datasets principal components and cross-tabbing them against each other.  I am assuming before you even get to collab filtering you are best served by carrying PCA on your data.  

My intuition on the relationship between PCA and Collab Filtering were wrong.  PCA seeks to find orthogonal components that maximize the variance.  In other words,  the characteristics that most contribute to the difference between each item in the sample.  Collab Filtering works on a completely different framework.

You can use multiple components in collaborative filtering matching users up with many different items

Collab filtering can be described as a User-Item paradigm, where by we are looking for the latent factors that describe a users behavior.  

# Learning the Latent Factors

Can use SGD to learn the latent factors. Init Params -> calc predictions -> calc loss -> backprop -> reset params

# Creating the DataLoaders

Always a good idea to make the data human interpretable. Add real words, values, etc.

Need to think about how to implement a dataloader myself.

DataLoaders for collab filtering should have a random splitter, needs to handle categorical/continuous data, A specific batched data format.

Should be able to parse through the training data the same as validation.  Build data partitioning into the dataloader.

for tabular data, should have a concept for classes in your data.  A dict with field name and the unique labels in that field. 

# Cross tabbing in Pytorch

Need to create tensors that store all of the parameters for each user and each item in your dataset that you wish to include in your analysis.

```
xfactors = torch.randn(unique len(user or item), number of factors)
```

Where x is any user/item in the data

# One hotting the factors

In order to access the proper factor pairs we could employ one-hotting , however, this would take up a lot of memory.  Most AI libraries implement an indexing function that has a derivative equal to the derivative on multiplying by a one hot vector.

We get the best of both worlds, so long as our memory efficient, one-hot analog function achieves the same output as the actual one-hot operation would then we can use the one-hot derivative since the two function are equivalent!

# Embeddings 

Embeddings are the learnable parameters attached to each user/item in the dataset of length n.  n is arbitrary but there are heuristics to pick good values for n.  

an embedding matrix is the cross tab of a user item pair with factors attached, the values of which are the dot products of each embedding pair.  

#  Creating the model from scratch

moved to a .py to build the model.  

When creating new Pytorch modules you must define a forward function that will called directly during the model training

The model input is a tensors with shape (batch_size, 2).  

Try to steal from the MNIST project to build in the training functionality, essentially build the naive learner.


After stealing some functionality I created a simple dataloader to use with the model.  The last piece of the puzzle is creating my own embeddings. I think this is covered in the chapter. 














