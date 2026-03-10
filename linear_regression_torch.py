import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


def train(X, y, lr=0.01, epochs=500):
    model = nn.Linear(1,1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    losses = []

    for epoch in range(500):

        y_pred = model(X)

        loss = torch.mean((y_pred - y) ** 2)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 50 == 0:
            print(epoch, loss.item())

    return model


# Forward pass. Prediction function. 
def predict(X, w, b):
    return X * w + b

# Loss function. 
def mse_loss(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

# Training setup.
lr = 0.01
epochs = 500

# Main structure. 
def main():
    # data
    X = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[2.0], [4.0], [6.0]])

    model = train(X, y)

    # plot predictions
    plt.scatter(X.numpy(), y.numpy(), label="Data")
    plt.plot(
        X.numpy(),
        model(X).detach().numpy(),
        color="red",
        label="Model",
    )
    plt.legend()
    plt.title("PyTorch Linear Regression")
    plt.show()


if __name__ == "__main__":
    main()        