import torch

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1)
    x_mps = x.to(device)
    print(x_mps * x_mps)

if __name__=="__main__":
    main()