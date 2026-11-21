import numpy as np
import math
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
        self.support = self.__calculateSupport(self.alpha, x, y)

        # 4. Bias
        self.bias = self.__calculateBias(self.alpha, self.kernel_value, y)

        # 5. Model (Decision Function)
        return self

    ## Fungsi Prediksi data klasifikasi
    def predict(self, x):
        score = self.__calculatePrediction(x)
        return 1 if score >= 0 else -1


    # Helper
    ## 1. Menghitung fungsi kernel untuk data x dan z
    def __kernel(self, xi,xj):
        if self.kernel == "linear":
            return np.dot(np.transpose(xi), xj)

        elif self.kernel == "polynomial":
            return pow((self.gamma * (np.dot(np.transpose(xi),xj)) + self.r), self.p)

        elif self.kernel == "gaussian":
            squared = np.sum(pow(np.abs(xi - xj), 2))
            return math.exp(-self.gamma * squared)

        else:
            raise ValueError("Unknown kernel")

    ## 2. Menghitung kernel untuk semua data
    def __computeKernel(self, x):
        xp = np.array(x)
        
        baris, kolom  = xp.shape
        kernel = np.zeros((baris,baris))
        for i in range(baris):
            for j in range (i,baris):
                value = self.__kernel(xp[i],xp[j])

                kernel[i][j] = value
                kernel[j][i] = value
        
        return kernel

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
        total = 0.0

        for i in range(len(self.support_vectors)):
            total += (
                self.support_alphas[i]
                * self.support_labels[i]
                * self.__kernel(self.support_vectors[i], x)
            )

        return total + self.bias
