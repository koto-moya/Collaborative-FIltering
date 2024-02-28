from modules.functions import func
import torch

class Trainer():
    def __init__(self, model, train_data, valid_data, wd = 0.1, lr = 0.005, epochs = 5):
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
            self.view()
        return self.model
    
    def train(self):
        for x,y in self.train_data:
            preds = self.model(x)
            self.loss = func.rmse(preds, y)
            self.loss.backward()
            self.update_params()

    def update_params(self):
        for param in self.model.parameters():
            param.grad += self.wd * param.data
            param.data -= self.lr*param.grad.data
            param.grad = None

    def validate_epoch(self):
        vl = [func.rmse(self.model(x),y) for x,y in self.valid_data]
        return round(torch.stack(vl).mean().item(), 4)
    
    def view(self):
        if self.current_epoch%100 == 0: 
            print(f"Loss: {self.loss} | ", f"Validate: {self.validate_epoch()} | ", f"Epoch: {self.current_epoch} | ", f"lr: {self.lr}")