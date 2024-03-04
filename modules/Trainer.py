from modules.functions import func
import torch
import time
import asyncio

class Trainer(): # "cpu")
    def __init__(self, model, train_data, valid_data, model_type='dot_prod', wd = 0.0, lr = 0.005, epochs = 5, loss_func=func.rmse):
        self.device = torch.device("cpu")#"mps" if torch.backends.mps.is_available() else "cpu")#"cpu") 
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.model_type = model_type
        self.lr = lr
        self.epochs = epochs
        self.loss = torch.tensor([[0]], device = self.device)
        self.wd = wd 
        self.loss_func = loss_func
    
    def train_loop(self):
        start_time=time.time()
        for i in range(self.epochs):
            self.current_epoch = i
            self.train()
            self.view()
            end_time = time.time()
            if i == 0:
                print(f"projected training time: {self.epochs*(end_time - start_time)/60}",  "mins")
            elif i%10 == 0:
                print(f"timer: {(end_time - start_time)/60}",  "mins")
        end_time = time.time()
        print(f"Total time elapsed: {(end_time - start_time)/60}",  "mins")
        return self.model
    
    def train(self):
        # make this async
        for x,y in self.train_data:
            preds = self.model(x)
            #print(preds)
            self.loss = self.loss_func(preds, y)
            self.loss.backward()
            self.update_params()

    def update_params(self):
        for param in self.model.parameters():
            param.grad += self.wd * param.data
            param.data -= self.lr*param.grad.data
            param.grad = None

    def dot_validate_epoch(self):
        vl = [self.loss_func(self.model(x),y) for x,y in self.valid_data]
        return round(torch.stack(vl).mean().item(), 4)
    
    def nn_validate_epoch(self):
        accs = [self.batch_accuracy(self.model(x),y) for x,y in self.valid_data]
        return round(torch.stack(accs).mean().item(), 4)
        
    def nn_batch_accuracy(self, x, y):
        preds = func.softmax(x)
        predicted_value = torch.argmax(preds, dim=1)
        print(predicted_value+1)
        trgts = y.flatten()
        bools = predicted_value+1 == trgts
        accs = bools.to(torch.float).mean()
        return accs
    
    def view(self):
        if self.model_type == 'dot_prod':
            if self.current_epoch%10 == 0: 
                print(f"Loss: {self.loss} | ", f"Validate: {self.dot_validate_epoch()} | ", f"Epoch: {self.current_epoch} | ", f"lr: {self.lr}") 
        else:
            if self.current_epoch%10 == 0: 
                print(f"Loss: {self.loss} | ", f"Validate: {self.nn_validate_epoch()} | ", f"Epoch: {self.current_epoch} | ", f"lr: {self.lr}")