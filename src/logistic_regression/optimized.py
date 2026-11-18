import numpy as np
from . import common

class LROptimized:
    PENALTIES = (None, 'l1', 'l2', 'elasticnet')
    SCHEDULES = ('constant', 'invscaling')
    add_bias = staticmethod(common.add_bias)
    sigmoid = staticmethod(common.sigmoid)

    def __init__(self, lr=0.05, shuffle=True, batch_size=256, penalty=None, alpha=0.01,
                 l1_ratio=0.5, learning_rate='constant', power_t=0.5, max_iter=1000,
                 tol=1e-7, iter_idle=20, class_weight=None, random_state=None,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        if penalty not in self.PENALTIES:
            raise ValueError(f"penalty harus salah satu dari {self.PENALTIES}")
        if learning_rate not in self.SCHEDULES:
            raise ValueError(f"learning_rate harus salah satu dari {self.SCHEDULES}")
        if alpha < 0 or not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("alpha >= 0 dan 0 <= l1_ratio <= 1")
        self.lr = lr
        self.batch_size = batch_size # 1 = SGD, k = mini-batch, None = full-batch
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
        self.beta1, self.beta2, self.eps = beta1, beta2, eps # Adam (adaptive moment estimation) optimizer params
        self.theta = None
        self.loss_history = []
        self.theta_history = []
        self.n_iter_ = 0

    def eta(self, epoch):
        return common.eta_value(self.lr, self.learning_rate, self.power_t, epoch)

    def sample_weights(self, y):
        return common.sample_weights(y, self.class_weight)

    def log_loss(self, X, y, w=None):
        return common.log_loss(self.theta, X, y, w)

    def penalty_loss(self):
        return common.penalty_loss(self.theta, self.penalty, self.alpha, self.l1_ratio)

    def data_gradient(self, X, y, w):
        return common.data_gradient(self.theta, X, y, w)

    def gradient(self, X, y, w=None):
        return common.gradient(self.theta, X, y, w, self.penalty, self.alpha, self.l1_ratio)

    def init_state(self, d):
        self.m = np.zeros(d)
        self.v = np.zeros(d)
        self.time = 0

    # Adam + Proximal Gradient Descent
    def step(self, X, y, w, eta):
        t = self.theta.copy()
        t[0] = 0.0
        grad = common.data_gradient(self.theta, X, y, w)
        if self.penalty in ('l2', 'elasticnet'):
            l2 = self.alpha * (1 - self.l1_ratio) if self.penalty == 'elasticnet' else self.alpha
            grad = grad + l2 * t

        self.time += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
        m_hat = self.m / (1 - self.beta1 ** self.time)
        v_hat = self.v / (1 - self.beta2 ** self.time)
        denom = np.sqrt(v_hat) + self.eps
        self.theta -= eta * m_hat / denom

        l1 = self.alpha * (self.l1_ratio if self.penalty == 'elasticnet' else 1.0)
        if self.penalty in ('l1', 'elasticnet'):
            thr = eta * l1 / denom
            thr[0] = 0.0                                     # bias tidak dipenalti
            self.theta = np.sign(self.theta) * np.maximum(np.abs(self.theta) - thr, 0.0)

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
        self.init_state(d)
        self.loss_history = []
        self.theta_history = []
        best_loss, best_theta, best_epoch, idle = np.inf, self.theta.copy(), -1, 0

        for epoch in range(self.max_iter):
            eta = self.eta(epoch)
            order = rng.permutation(n) if self.shuffle else np.arange(n)
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                self.step(X[idx], y[idx], w[idx], eta)

            loss = self.log_loss(X, y, w) + self.penalty_loss()
            self.loss_history.append(loss)
            self.theta_history.append(self.theta.copy())
            self.n_iter_ = epoch + 1
            if loss < best_loss - self.tol:
                best_loss, best_theta, best_epoch, idle = loss, self.theta.copy(), epoch, 0
            else:
                idle += 1
                if idle >= self.iter_idle:
                    break
        self.theta = best_theta                              # kembalikan bobot terbaik
        self.theta_history = self.theta_history[:best_epoch + 1]
        return self

    def decision_function(self, X):
        return common.add_bias(X) @ self.theta

    def predict_proba(self, X):
        return common.predict_proba(self.theta, X)

    def predict(self, X, threshold=0.5):
        return common.predict(self.theta, X, threshold)

    def score(self, X, y):
        return common.score(self.theta, X, y)

    @property
    def n_nonzero_(self):
        return common.nonzero_count(self.theta)

    def get_params(self):
        return common.get_params(self)

    def save(self, path):
        common.save_model(path, self.theta, self.loss_history, self.theta_history, self.get_params())

    @classmethod
    def load(cls, path):
        theta, loss_history, theta_history, params = common.load_model(path)
        model = cls(**params)
        model.theta = theta
        model.loss_history = loss_history
        model.theta_history = theta_history
        return model