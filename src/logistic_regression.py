"""
Owner E — Logistic regression from scratch (NumPy only)
See notebooks/03_logistic_regression.ipynb for training and evaluation.
"""
import numpy as np
import pickle


class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, tol=1e-5):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _compute_loss(self, y, y_hat, sample_weights):
        eps = 1e-12
        loss = -sample_weights * (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        return loss.mean()

    def fit(self, X, y, class_weight="balanced"):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        # compute per-sample weights
        sample_weights = self._get_sample_weights(y, class_weight)

        prev_loss = np.inf
        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            loss = self._compute_loss(y, y_hat, sample_weights)
            self.loss_history.append(loss)

            error = sample_weights * (y_hat - y)
            dw = (X.T @ error) / n_samples
            db = error.mean()

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self

    def _get_sample_weights(self, y, class_weight):
        if class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            weight_map = {c: n_samples / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        elif isinstance(class_weight, dict):
            weight_map = class_weight
        else:
            return np.ones(len(y))

        return np.array([weight_map[label] for label in y])

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "bias": self.bias}, f)

    def load(self, path):
        with open(path, "rb") as f:
            params = pickle.load(f)
        self.weights = params["weights"]
        self.bias = params["bias"]
        self.threshold = params.get("threshold", 0.5)
        return self
