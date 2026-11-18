import inspect
import json
import numpy as np

def add_bias(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X harus 2 dimensi (n_samples, n_features)")
    return np.hstack([np.ones((X.shape[0], 1)), X])

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def eta_value(lr, learning_rate, power_t, epoch):
    if learning_rate == 'constant':
        return lr
    return lr / (epoch + 1) ** power_t

# bobot per contoh, dinormalisasi ke rata-rata 1 agar skala lr tidak berubah
def sample_weights(y, class_weight):
    w = np.ones_like(y)
    cw = class_weight
    if cw == 'balanced':
        n, n1 = len(y), y.sum()
        cw = None if n1 in (0, n) else {0: n / (2 * (n - n1)), 1: n / (2 * n1)}
    if cw is not None:
        w = np.where(y == 1, cw.get(1, 1.0), cw.get(0, 1.0)).astype(float)
    return w / w.mean()

# mean weighted log loss, X sudah berisi kolom bias
def log_loss(theta, X, y, w=None):
    z = X @ theta
    losses = np.logaddexp(0.0, z) - y * z
    return float(np.mean(losses if w is None else w * losses))

def penalty_loss(theta, penalty, alpha, l1_ratio):
    t = theta[1:]
    l1 = np.abs(t).sum()
    l2 = 0.5 * np.dot(t, t)
    if penalty == 'l1':
        return alpha * l1
    if penalty == 'l2':
        return alpha * l2
    if penalty == 'elasticnet':
        return alpha * (l1_ratio * l1 + (1 - l1_ratio) * l2)
    return 0.0

def data_gradient(theta, X, y, w=None):
    w = np.ones_like(y) if w is None else w
    p = sigmoid(X @ theta)
    return X.T @ (w * (p - y)) / len(y)

def penalty_grad(theta, penalty, alpha, l1_ratio):
    t = theta.copy()
    t[0] = 0.0 # bias tidak dipenalti
    if penalty == 'l1':
        return alpha * np.sign(t)
    if penalty == 'l2':
        return alpha * t
    if penalty == 'elasticnet':
        return alpha * (l1_ratio * np.sign(t) + (1 - l1_ratio) * t)
    return np.zeros_like(t)

# gradien mean log loss + penalti pada batch, X sudah berisi kolom bias
def gradient(theta, X, y, w, penalty, alpha, l1_ratio):
    return data_gradient(theta, X, y, w) + penalty_grad(theta, penalty, alpha, l1_ratio)

def predict_proba(theta, X):
    return sigmoid(add_bias(X) @ theta)

def predict(theta, X, threshold=0.5):
    return (predict_proba(theta, X) >= threshold).astype(int)

def score(theta, X, y, threshold=0.5):
    return float(np.mean(predict(theta, X, threshold) == np.asarray(y).ravel()))

def nonzero_count(theta):
    return int(np.count_nonzero(theta[1:]))

def get_params(instance):
    names = set(inspect.signature(type(instance).__init__).parameters) - {'self'}
    return {name: getattr(instance, name) for name in sorted(names)}

def save_model(path, theta, loss_history, theta_history, params):
    np.savez(path, theta=theta, loss_history=np.array(loss_history), theta_history=np.array(theta_history), params=json.dumps(params))

def load_model(path):
    data = np.load(str(path) if str(path).endswith('.npz') else f"{path}.npz")
    params = json.loads(str(data['params']))
    if isinstance(params.get('class_weight'), dict):     # kunci JSON -> int
        params['class_weight'] = {int(k): v for k, v in params['class_weight'].items()}
    return data['theta'], data['loss_history'].tolist(), data['theta_history'].tolist(), params