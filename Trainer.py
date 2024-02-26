from fastai.collab import *
from fastai.tabular.all import *

class Trainer():
    def __init__(self, model, train_data, valid_data, wd = 0.0, lr = 0.003, epochs = 100):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.lr = lr
        self.epochs = epochs
        self.loss = torch.tensor([[0]])
        self.wd = wd 
    
    def train_loop(self):
        for i in range(self.epochs):
            self.current_epoch = i
            self.train()
    
    def train(self):
        for x,y in self.train_data:
            preds = self.model(x)
            self.loss = self.rmse(preds, y)
            self.loss.backward()
            self.update_params()

    def update_params(self):
        for param in self.model.parameters():
            # param.grad += self.wd * param.data for implementing weight decay??
            param.data -= self.lr*param.grad.data
            param.grad = None

    def rmse(self, preds, y):
        return ((preds - y)**2).mean().sqrt()
    
    def softmax(self, preds):
        preds = preds-torch.max(preds)
        return torch.exp(preds)/torch.sum(torch.exp(preds), dim=1).unsqueeze(1)
    
    def cross_entropy_loss(self, preds, trgt):
        log_soft = torch.log(self.softmax(preds))
        one_hot = log_soft[range(len(log_soft)),trgt.flatten()]
        cel = -torch.sum(one_hot)
        return cel

    def validate_epoch(self):
        accs = [self.batch_accuracy(self.model(x),y) for x,y in self.valid_dl]
        return round(torch.stack(accs).mean().item(), 4)
        
    def batch_accuracy(self, x, y):
        preds = self.softmax(x)
        predicted_value = torch.argmax(preds, dim=1)
        trgts = y.flatten()
        bools = predicted_value == trgts
        accs = bools.to(torch.float).mean()
        return accs
    
    def accuracy(self):
        return torch.stack([self.batch_accuracy(self.softmax(self.model(x)), y) for x,y in self.valid_data]).mean()
