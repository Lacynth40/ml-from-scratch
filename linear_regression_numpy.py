import numpy as np
import matplotlib.pyplot as plt

# NumPy role: EXPLAIN THE MATH



def load_data():
    """
    Returns:
        X: (n_samples, n_features)
        y: (n_samples,)
    """
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    return X, y

def predict(X, weights):
    # X @ weights = multiply inputs by weights and sum per row
    # return X @ weights
    w, b = weights
    return X.flatten() * w + b

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradient(X, y, y_pred):
    error = y_pred - y
    dw = np.mean(2 * X.flatten() * error)
    db = np.mean(2 * error)
    return np.array([dw, db])

def train(X, y, lr=0.01, epochs=500):
    weights = np.array([0.0, 0.0])  # [w, b]
    losses = []
    for epoch in range(epochs):
        y_pred = predict(X, weights)
        loss = mse_loss(y, y_pred)
        grads = compute_gradient(X, y, y_pred)

        weights -= lr * grads
        losses.append(loss)

        if epoch % 50 == 0:
            print(f"epoch {epoch} | loss {loss:.4f} | w {weights[0]:.4f} | b {weights[1]:.4f}")

    return weights, losses

def main():
    X, y = load_data()
    weights, losses = train(X, y)

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.show()

    # scatter original data
    plt.scatter(X, y, label="Data")

    # predicted line
    y_line = predict(X, weights)
    plt.plot(X, y_line, color="red", label="Model")

    plt.legend()
    plt.title("Linear Regression Fit")
    plt.show()

if __name__ == "__main__":
    main()