import torch
import matplotlib.pyplot as plt


def train(X, y, lr=0.01, epochs=500):
    # Define the parameters.
    w = torch.tensor([0.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    losses = []

    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        loss = mse_loss(y_pred, y)

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

        losses.append(loss.item())

        if epoch % 50 == 0:
            print(
                f"epoch {epoch} | loss {loss.item():.4f} | "
                f"w {w.item():.4f} | b {b.item():.4f}"
            )

    return w, b, losses

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

    w, b, losses = train(X, y)

    # plot predictions
    plt.scatter(X.numpy(), y.numpy(), label="Data")
    plt.plot(
        X.numpy(),
        predict(X, w, b).detach().numpy(),
        color="red",
        label="Model",
    )
    plt.legend()
    plt.title("PyTorch Linear Regression")
    plt.show()


if __name__ == "__main__":
    main()        