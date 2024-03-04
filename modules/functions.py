import torch

class func:
    @staticmethod
    def sigmoid_range(x, low, high):
        return torch.sigmoid(x) * (high-low) + low

    @staticmethod
    def mse(preds, y):
        return ((preds - y)**2).mean()
    
    @staticmethod
    def rmse(preds, y):
        return func.mse(preds, y).sqrt()
    
    @staticmethod
    def softmax(preds):
        preds = preds-torch.max(preds)
        return torch.exp(preds)/torch.sum(torch.exp(preds), dim=1).unsqueeze(1)
    
    @staticmethod
    def cross_entropy_loss(preds, y):
        log_soft = torch.log(func.softmax(preds))
        one_hot = log_soft[range(len(log_soft)),y.flatten()-1]
        cel = -torch.sum(one_hot)
        return cel
    