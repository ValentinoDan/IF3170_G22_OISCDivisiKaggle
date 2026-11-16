import inspect
import json
import numpy as np

class LogisticRegression:
    PENALTIES = (None, 'l1', 'l2', 'elasticnet')
    SCHEDULES = ('constant', 'invscaling')

    def __init__(self, lr=0.1, shuffle=True, batch_size=1, penalty=None, alpha=0.01,
                 l1_ratio=0.5, learning_rate='constant', power_t=0.5, max_iter=1000,
                 tol=1e-4, iter_idle=5, class_weight=None, random_state=None):
        if penalty not in self.PENALTIES:
            raise ValueError(f"penalty harus salah satu dari {self.PENALTIES}")
        if learning_rate not in self.SCHEDULES:
            raise ValueError(f"learning_rate harus salah satu dari {self.SCHEDULES}")
        if alpha < 0 or not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("alpha >= 0 dan 0 <= l1_ratio <= 1")
        self.lr = lr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.max_iter = max_iter
        self.tol = tol
        self.iter_idle = iter_idle
        self.class_weight = class_weight # None | 'balanced' | {0: w0, 1: w1}
        self.random_state = random_state
        self.theta = None
        self.loss_history = []
        self.n_iter_ = 0

    @staticmethod
    def add_bias(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X harus 2 dimensi (n_samples, n_features)")
        return np.hstack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def sigmoid(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def eta(self, epoch):
        if self.learning_rate == 'constant':
            return self.lr
        return self.lr / (epoch + 1) ** self.power_t

    # bobot per contoh, dinormalisasi ke rata-rata 1 agar skala lr tidak berubah
    def sample_weights(self, y):
        w = np.ones_like(y)
        cw = self.class_weight
        if cw == 'balanced':
            n, n1 = len(y), y.sum()
            cw = None if n1 in (0, n) else {0: n / (2 * (n - n1)), 1: n / (2 * n1)}
        if cw is not None:
            w = np.where(y == 1, cw.get(1, 1.0), cw.get(0, 1.0)).astype(float)
        return w / w.mean()

    # mean log loss, X sudah berisi kolom bias
    def log_loss(self, X, y, w=None):
        z = X @ self.theta
        losses = np.logaddexp(0.0, z) - y * z
        return float(np.mean(losses if w is None else w * losses))

    def penalty_loss(self):
        t = self.theta[1:]
        l1 = np.abs(t).sum()
        l2 = 0.5 * np.dot(t, t)
        if self.penalty == 'l1':
            return self.alpha * l1
        if self.penalty == 'l2':
            return self.alpha * l2
        if self.penalty == 'elasticnet':
            return self.alpha * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)
        return 0.0
    
    # gradien mean log loss + penalti pada batch, X sudah berisi kolom bias
    def gradient(self, X, y, w=None):
        w = np.ones_like(y) if w is None else w
        p = self.sigmoid(X @ self.theta)
        grad = X.T @ (w * (p - y)) / len(y)
        t = self.theta.copy()
        t[0] = 0.0 # bias tidak dipenalti
        if self.penalty == 'l1':
            grad += self.alpha * np.sign(t)
        elif self.penalty == 'l2':
            grad += self.alpha * t
        elif self.penalty == 'elasticnet':
            grad += self.alpha * (self.l1_ratio * np.sign(t) + (1 - self.l1_ratio) * t)
        return grad

    def fit(self, X, y):
        X = self.add_bias(X)
        y = np.asarray(y, dtype=float).ravel()
        if not np.all(np.isin(y, (0, 1))):
            raise ValueError("y harus biner {0,1}")
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        batch = n if self.batch_size is None else min(self.batch_size, n)
        w = self.sample_weights(y)

        self.theta = np.zeros(d)
        self.loss_history = []
        best_loss, best_theta, idle = np.inf, self.theta.copy(), 0

        for epoch in range(self.max_iter):
            eta = self.eta(epoch)
            order = rng.permutation(n) if self.shuffle else np.arange(n)
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                self.theta -= eta * self.gradient(X[idx], y[idx], w[idx])

            loss = self.log_loss(X, y, w) + self.penalty_loss()
            self.loss_history.append(loss)
            self.n_iter_ = epoch + 1
            if loss < best_loss - self.tol:
                best_loss, best_theta, idle = loss, self.theta.copy(), 0
            else:
                idle += 1
                if idle >= self.iter_idle:
                    break
        self.theta = best_theta
        return self

    def decision_function(self, X):
        return self.add_bias(X) @ self.theta

    def predict_proba(self, X):
        return self.sigmoid(self.decision_function(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    @property
    def n_nonzero_(self):
        return int(np.count_nonzero(self.theta[1:]))

    def get_params(self):
        names = set(inspect.signature(type(self).__init__).parameters) - {'self'}
        return {name: getattr(self, name) for name in sorted(names)}

    def save(self, path):
        np.savez(path, theta=self.theta, loss_history=np.array(self.loss_history), params=json.dumps(self.get_params()))

    @classmethod
    def load(cls, path):
        data = np.load(str(path) if str(path).endswith('.npz') else f"{path}.npz")
        params = json.loads(str(data['params']))
        if isinstance(params.get('class_weight'), dict):
            params['class_weight'] = {int(k): v for k, v in params['class_weight'].items()}
        model = cls(**params)
        model.theta = data['theta']
        model.loss_history = data['loss_history'].tolist()
        return model