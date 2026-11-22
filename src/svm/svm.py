import numpy as np
from qpsolvers import solve_qp 

# Implementasi dari SVM
class SVM:
    # Pilihan kernel: "linear", "polynomial", "gaussian"
    def __init__(self, gamma = 1.0, r = 0.0, p = 1, c = 1, kernel = "linear"):
        self.kernel = kernel
        self.gamma = gamma
        self.r = r 
        self.p = p
        self.c = c

    # Main Method
    ## Fungsi Training Data
    def fit(self, x, z):
        # Asumsi: data sudah valid
        # x : x_train
        # z : y_train 
        x = np.asarray(x, dtype=float)
        y = np.asarray(z, dtype=float).reshape(-1)

        # 1. Hitung Kernel
        self.kernel_value = self.__computeKernel(x)

        # 2. Solve QP 
        self.alpha = self.__solveQP(self.kernel_value, y)

        # 3. Support Vector 
        self.__calculateSupport(self.alpha, x, y)

        # 4. Bias
        self.bias = self.__calculateBias(self.alpha, self.kernel_value, y)

        # 5. Model (Decision Function)
        return self

    ## Fungsi Prediksi data klasifikasi
    def predict(self, x):
        score = self.__calculatePrediction(x)

        if (np.ndim(score) == 0):
            return 1 if score >= 0 else -1
        return np.where(score>=0,1,-1)


    # Helper
    ## 1. Menghitung fungsi kernel 
    def __kernel(self, A, B):
        dot = A @ B.T
        if self.kernel == "linear":
            return dot

        elif self.kernel == "polynomial":
            return (self.gamma * dot + self.r) ** self.p

        elif self.kernel == "gaussian":
            a_sq = np.einsum("ij,ij->i", A, A)[:, None]
            b_sq = np.einsum("ij,ij->i", B, B)[None, :]
            squared = np.maximum(a_sq + b_sq - 2.0 * dot, 0.0) 
            return np.exp(-self.gamma * squared)

        else:
            raise ValueError("Unknown kernel")

    ## 2. Menghitung kernel untuk semua data
    def __computeKernel(self, x):
        return self.__kernel(x,x)

    ## 3. Menghitung Alpha
    def __solveQP(self, kernel, y):
        n = len(y)

        # 1
        P = np.outer(y, y) * kernel
        q = -np.ones(n)

        # 2
        A = y.reshape(1, -1)
        b = np.array([0.0])

        # 3 (bounds)
        lb = np.zeros(n)
        ub = np.full(n, self.c)

        # 4 (Solve QP Problem)
        alpha = solve_qp(
            P=P, q=q, A=A, b=b, lb=lb, ub=ub, solver="osqp",
            eps_abs=1e-6, eps_rel=1e-6, max_iter=10000000,
        )

        # (Validasi)
        if alpha is None:
            raise RuntimeError("Alpha Failed")
        
        return alpha

    ## 4. Menghitung Support Vector
    def __calculateSupport(self, alpha, X, y):
        # Batas untuk angka yang tidak pure 0
        batas = 1e-6
        condition = alpha > batas

        index = np.zeros(len(X))

        self.support_vectors = X[condition]
        self.support_labels = y[condition]
        self.support_alphas = alpha[condition]
        self.support_index = np.where(condition)[0]

        self.dual_coef = self.support_alphas * self.support_labels
        self.support_norm_sq = np.einsum("ij,ij->i",self.support_vectors, self.support_vectors)
        # Khusus Linear
        if self.kernel == "linear":
            self.w = (self.dual_coef @ self.support_vectors) 


    ## 5. Menghitung bias
    def __calculateBias(self, alpha, kernel, y):
        # Batas untuk angka yang tidak pure 0
        batas = 1e-6
        condition = (alpha>batas) & (alpha<self.c-batas)

        if not np.any(condition):
            raise ValueError("No support vector")

        totals = kernel[condition, :] @ (alpha * y)
        
        bias = y[condition] - totals

        return np.mean(bias) 
        
    ## 6. Fungsi untuk menjalankan perhitungan prediksi
    def __calculatePrediction(self, x):
        x = np.asarray(x, dtype=float)

        if (self.kernel == "linear"):
            return x @ self.w + self.bias

        x2d = np.atleast_2d(x)
        result = self.__kernel(x2d, self.support_vectors) @ self.dual_coef + self.bias

        if (x.ndim == 1):
            return float(result[0])
        else:
            return result 
