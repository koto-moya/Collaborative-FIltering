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

working in parallel with the notebook as I learn more.  building out the model class in the notebook to progress through to embeddings. 

In these models, its a good idea to squash the outputs into the range of values present in data.  For example, if you are predicting movie ratings that range 1-5 you should only be predicting values that occur on this range.

You can achieve this through a scaled sigmoid but one thing need to be taken into account.  A unit sigmoid will never actually reach 1, it will only get asymptotically close to it.  Therefore, you should make your range extend a little further than the highest value so you can actually reach it.  

Introduced a sigmoid which helped the training slightly but the model overfit at the end

Introducing a bias parameter also did not help.  The model overfits, this is where Weight Decay comes into play. 

# Weight Decay

Implementing weight decay with fastai libraries is pretty simple just add the `wd` arg.  

Implementing in my model will take just a few steps of motivation.  

Weight decay, in it's simplest is a factor included in the loss that penalizes large weight more harshly than small weights.  

In psuedocode: `loss_with_wd = loss + l*(parameters**2).sum()`

so, we can see that due to the squaring of values, large parameters are penalized much more harshly than smaller ones.

Through some basic math, we can more efficiently introduce the weight decay param by adding it directly to the gradient, taking the derivative of the weight decay term leading to 

`parameter.grad += wd * 2* parameters`'

even further, using my favorite abuses of mathematics, we can just roll up the 2 into wd since its a parameter we can choose.  


so this becomes 

`parameter.grad += wd * parameters`

So to implement this into my model I think I can literally add the above line into my parameter update function.  

Haven't tested it yet but will once I implement the embedding function. 

# Creating your own Embedding Module

just need a way to create tensor of a certain shape with randomly initialized parameters.  Another constraint is that we need this tensor to be tracked by torch for use in calculating gradients. 

After creating the embedding function my model was completely free of dependencies.

# Training my Model

I had to include a few updates to the model to get it to train at all.  First, I updated the DataLoader to output the batches in a format that the model can use.   Then, I updated the trainer to use RMSE for the Loss as well as the validation metrics.  Finally, I implemented weight decay with `param.grad += self.wd * param.data` with `wd` as an init parameter.  

Training the model resulted in some interesting outcomes.  At first the model trains well, Loss and validation moving down in lock step. After about 60 epochs the model converges with hyperparameters: (factors=50, lr=0.01, wd = 0.01, epochs=100).  

**Tests**
hyperparameters: (factors=100, lr=0.001 -> 0.01 @ 250, wd = 0.001, epochs=300)

Results: Loss: 0.9073621034622192 |  Validate: 0.9914 |  Epoch: 299 |  lr: 0.01
---------------
hyperparameters: (factors=100, lr=0.001 -> 0.01 @ 250, wd = 0.001, epochs=500))

Results: Loss: 0.8239756226539612 |  Validate: 0.9707 |  Epoch: 490 |  lr: 0.01
---------------
hyperparamters: (factors=100, lr=0.001, wd = 0.001, epochs=500)

Results: Loss: 1.0798165798187256 |  Validate: 0.995 |  Epoch: 490 |  lr: 0.001
---------------
hyperparameters: (factors=100, lr=0.01, wd = 0.001, epochs=500)

Results: Loss: 0.9162543416023254 |  Validate: 0.9672 |  Epoch: 490 |  lr: 0.01
---------------
hyperparameters: (factors=100, lr=0.1, wd = 0.001, epochs=500)

Results: Loss: 0.8993635177612305 |  Validate: 0.9697 |  Epoch: 220 |  lr: 0.1 Converged at epoch 40
---------------
hyperparameters:(factors=100, lr=0.02, wd = 0.001, epochs=500)

Results: Loss: 1.0190260410308838 |  Validate: 0.9697 |  Epoch: 230 |  lr: 0.02
---------------
hyperparameters: (factors=100, lr=0.02, wd = 0.00001, epochs=500)

Results: Loss: 0.630207359790802 |  Validate: 0.9234 |  Epoch: 490 |  lr: 0.02
---------------
hyperparameters: (factors=100, lr=0.02, wd = 0.00001, epochs=500)

Results: Loss: 0.1812155693769455 |  Validate: 0.9876 |  Epoch: 490 |  lr: 0.03 Diverged at epoch 310

Results at Epoch 310: Loss: 0.6833301186561584 |  Validate: 0.9081 |  Epoch: 300 |  lr: 0.03
---------------
hyperparameters:(bs=64,  factors=100, lr=0.018 wd = 0.00001, epochs=500)

Results: Loss: 0.8128405809402466 |  Validate: 0.9033 |  Epoch: 499 |  lr: 0.018

Winner was lr=0.018 at a wd of 0.00001 for 500 epochs


# Saving the model

I pickled the model to save for future use.  

# Updating architecture

I wanted to relieve the project of the Fastai dependency which was successful (only kept it in main to download the data, I'm sure I can find it elsewhere, this is just easier for now).  Now the architecture relies solely on torch.  Doing this invalidated the first pickled model.  I reran the model under the new setup and pickled the resulting model.  

# Interpreting Embedding and Bias

so my model had one quirk that I couldn't quite understand.  I needed to offset the input tensor values by to correctly index the factors tensor.  The example in the book did not do this, for a while I couldn't understand why.  I knew the answer the whole time, when I printed the users form the fastai notebook there was always a '#na' value.  I couldn't find the value in the dataset however.  This confused me, but I moved on since I realized I could just offset the tensor.  It was only until I started playing with the biases and embeddings that I realized this would be a problem.  I guess I just did not expect the .classes function to add a data point to the data.  Going to add this in so that my architecture more closely matches that of the book.  

Ended up fixing the model, it is still slightly different than the model in the book but I'll take it.  

# Transfer the model to a Deep Learning Architecture

I decided not to do this portion since I was happy with how the Collab model turned out. 

Now the next challenge is to get this model to run on mps which might melt my brain if I get the Conv2dPooling error again.

# Running on MPS

Okay so it turns out that MPS is kinda trash for this model.  I think it does better with larger batch sizes or just larger workloads in general.  Tabling this for now, the CPU performance is more than enough for what I need.  ~5x faster on the CPU.

# More Optimization

I need to start comparing the validation metric (rmse) with the standard deviation and variance of the dataset to get an idea of performance.  






















