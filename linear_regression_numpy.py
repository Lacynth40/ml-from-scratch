import numpy as np
import matplotlib.pyplot as plt

# NumPy role: EXPLAIN THE MATH



def load_data():
    """
    Returns:
        X: (n_samples, n_features)
        y: (n_samples,)
    """
    # TODO: generate or load simple linear data
    print("load_data works")
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    return X, y

def predict(X, weights):
    # X @ weights = multiply inputs by weights and sum per row
    print("predict called")
    return X @ weights

def mse_loss(y_true, y_pred):
    print("mse_loss called")
    return np.mean((y_true - y_pred) ** 2)

def compute_gradients(X, y, y_pred):
    print("compute_gradient called")
    return np.mean(2 * X.flatten() * (y_pred - y))

def train(X, y, lr=0.01, epochs=500):
    # TODO:
    # 1. initialize weights
    # 2. loop epochs
    # 3. forward pass
    # 4. loss
    # 5. gradients
    # 6. update weights
    print("train works")
    weights = np.array([0.0])   # fake placeholder
    losses = []                 # empty list for now
     # NEW: run exactly once
    for _ in range(1):
        y_pred = predict(X, weights)
        loss = mse_loss(y, y_pred)
        grad = compute_gradients(X, y, y_pred)

        print("loss:", loss)
        print("gradient:", grad)

        weights[0] -= lr * grad
        print("updated weight:", weights[0])
    return weights, losses
def main():
    X, y = load_data()
    weights, losses = train(X, y)

    plt.plot(losses)
    plt.title("NumPy Training Loss")
    plt.show()

if __name__ == "__main__":
    main()