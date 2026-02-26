import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# PyTorch role: EXECUTE PROFESSIONALLY

def load_data():
    # TODO: same data as NumPy version (torch tensors)
    pass

def main():
    X, y = load_data()

    # NumPy weights → torch.nn.Linear
    model = nn.Linear(X.shape[1], 1)

    # NumPy MSE → nn.MSELoss
    criterion = nn.MSELoss()

    # NumPy gradient descent → torch.optim
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = []

    for _ in range(500):
        # TODO:
        # zero gradients
        # forward pass
        # loss
        # backward pass
        # optimizer step
        pass

    plt.plot(losses)
    plt.title("PyTorch Training Loss")
    plt.show()

if __name__ == "__main__":
    main()