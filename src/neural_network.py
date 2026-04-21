"""
Owner F — Neural network from scratch (NumPy only)
See notebooks/04_neural_network.ipynb for the full implementation.
"""
import pickle
import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        hidden_layers=(32,),
        lr=0.01,
        n_iter=1000,
        batch_size=None,
        tol=1e-5,
        l2=0.0,
        random_state=42,
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol
        self.l2 = l2
        self.random_state = random_state
        self.params = {}
        self.loss_history = []
        self.threshold = 0.5

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_grad(z):
        return z > 0.0

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.random_state)
        layer_sizes = (n_features, *self.hidden_layers)
        self.params = {}

        for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), start=1):
            self.params[f"W{i}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            self.params[f"b{i}"] = np.zeros(fan_out)

        out_fan_in = layer_sizes[-1]
        self.params["W_out"] = rng.normal(0.0, np.sqrt(1.0 / out_fan_in), size=(out_fan_in, 1))
        self.params["b_out"] = np.zeros(1)
        self.params["n_layers"] = len(self.hidden_layers)

    def _get_sample_weights(self, y, class_weight):
        if class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            weights = {c: n_samples / (len(classes) * count) for c, count in zip(classes, counts)}
            return np.array([weights[label] for label in y], dtype=float)
        if isinstance(class_weight, dict):
            return np.array([class_weight[label] for label in y], dtype=float)
        return np.ones(len(y), dtype=float)

    def _forward(self, X):
        activations = [X]
        pre_activations = []

        a = X
        for i in range(1, self.params["n_layers"] + 1):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            pre_activations.append(z)
            a = self._relu(z)
            activations.append(a)

        z_out = a @ self.params["W_out"] + self.params["b_out"]
        pre_activations.append(z_out)
        y_hat = self._sigmoid(z_out)
        activations.append(y_hat)
        return activations, pre_activations

    def _compute_loss(self, y, y_hat, sample_weights):
        eps = 1e-12
        y_col = y.reshape(-1, 1)
        weights_col = sample_weights.reshape(-1, 1)
        data_loss = -weights_col * (
            y_col * np.log(y_hat + eps) + (1.0 - y_col) * np.log(1.0 - y_hat + eps)
        )
        loss = data_loss.mean()
        if self.l2:
            weight_sum = sum(np.sum(v * v) for k, v in self.params.items() if k.startswith("W"))
            loss += 0.5 * self.l2 * weight_sum
        return loss

    def _backward(self, y, sample_weights, activations, pre_activations):
        n_samples = y.shape[0]
        y_col = y.reshape(-1, 1)
        weights_col = sample_weights.reshape(-1, 1)
        grads = {}

        dz = weights_col * (activations[-1] - y_col)
        grads["W_out"] = (activations[-2].T @ dz) / n_samples + self.l2 * self.params["W_out"]
        grads["b_out"] = dz.mean(axis=0)
        da = dz @ self.params["W_out"].T

        for i in range(self.params["n_layers"], 0, -1):
            dz = da * self._relu_grad(pre_activations[i - 1])
            grads[f"W{i}"] = (activations[i - 1].T @ dz) / n_samples + self.l2 * self.params[f"W{i}"]
            grads[f"b{i}"] = dz.mean(axis=0)
            if i > 1:
                da = dz @ self.params[f"W{i}"].T

        return grads

    def fit(self, X, y, class_weight="balanced"):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(float).ravel()
        self._init_params(X.shape[1])
        sample_weights = self._get_sample_weights(y, class_weight)
        batch_size = len(y) if self.batch_size is None else min(self.batch_size, len(y))
        rng = np.random.default_rng(self.random_state)
        prev_loss = np.inf
        self.loss_history = []

        for _ in range(self.n_iter):
            if batch_size == len(y):
                batches = (np.arange(len(y)),)
            else:
                batches = np.array_split(rng.permutation(len(y)), range(batch_size, len(y), batch_size))

            for idx in batches:
                activations, pre_activations = self._forward(X[idx])
                grads = self._backward(y[idx], sample_weights[idx], activations, pre_activations)
                for key, grad in grads.items():
                    self.params[key] -= self.lr * grad

            y_hat = self._forward(X)[0][-1]
            loss = self._compute_loss(y, y_hat, sample_weights)
            self.loss_history.append(loss)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._forward(X)[0][-1].ravel()

    def predict(self, X, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.params | {"threshold": self.threshold}, f)

    def load(self, path):
        with open(path, "rb") as f:
            params = pickle.load(f)
        self.threshold = params.pop("threshold", 0.5)
        self.params = params
        self.hidden_layers = tuple(params[f"W{i}"].shape[1] for i in range(1, params["n_layers"] + 1))
        return self
