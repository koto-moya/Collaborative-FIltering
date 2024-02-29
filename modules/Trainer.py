from modules.functions import func
import torch
import time

class Trainer(): # "cpu")
    def __init__(self, model, train_data, valid_data, wd = 0.1, lr = 0.005, epochs = 5, loss_func=func.rmse):
        self.device = torch.device("cpu")#"mps" if torch.backends.mps.is_available() else "cpu") 
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
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
        print((end_time - start_time)/60,  "mins")
        return self.model
    
    def train(self):
        for x,y in self.train_data:
            preds = self.model(x)
            self.loss = self.loss_func(preds, y)
            self.loss.backward()
            self.update_params()

    def update_params(self):
        for param in self.model.parameters():
            param.grad += self.wd * param.data
            param.data -= self.lr*param.grad.data
            param.grad = None

    def validate_epoch(self):
        vl = [self.loss_func(self.model(x),y) for x,y in self.valid_data]
        return round(torch.stack(vl).mean().item(), 4)
    
    def view(self):
        if self.current_epoch%100 == 0: 
            print(f"Loss: {self.loss} | ", f"Validate: {self.validate_epoch()} | ", f"Epoch: {self.current_epoch} | ", f"lr: {self.lr}")