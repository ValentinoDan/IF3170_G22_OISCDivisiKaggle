import numpy as np

def solve_smo(y, c, kernel_row, kernel_diag, tol=1e-6, max_passes=5, max_sweeps=100, random_state=42,):
    y = np.asarray(y, dtype=float).reshape(-1)
    kernel_diag = np.asarray(kernel_diag, dtype=float).reshape(-1)
    n = len(y)
    C = float(c)

    if n < 2:
        raise ValueError("Jumlah data training minimal 2 untuk SMO.")

    alpha = np.zeros(n, dtype=float)
    b = 0.0

    # f_raw[k] = sum_m alpha_m * y_m * K(x_m, x_k), sehingga f(x_k) = f_raw[k] + b
    # Di-update secara inkremental O(N) setiap kali (alpha_i, alpha_j) berubah
    f_raw = np.zeros(n, dtype=float)

    rng = np.random.default_rng(random_state)
    passes = 0
    sweeps = 0

    while passes < max_passes and sweeps < max_sweeps:
        num_changed = 0
        sweeps += 1

        for i in range(n):
            # 1. Hitung error E_i = f(x_i) - y_i
            E_i = (f_raw[i] + b) - y[i]

            # 2. Cek apakah sampel i melanggar kondisi KKT
            if (y[i] * E_i < -tol and alpha[i] < C) or (y[i] * E_i > tol and alpha[i] > 0.0):

                # 3. Pilih indeks j != i secara acak
                j = int(rng.integers(0, n - 1))
                if j >= i:
                    j += 1

                E_j = (f_raw[j] + b) - y[j]

                ai_old = alpha[i]
                aj_old = alpha[j]

                # 4. Hitung batas kotak [L, H] untuk alpha_j
                if y[i] != y[j]:
                    L = max(0.0, aj_old - ai_old)
                    H = min(C, C + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - C)
                    H = min(C, ai_old + aj_old)

                if L == H:
                    continue

                # 5. Ambil baris kernel K_i dan hitung kurvatur eta
                K_i = kernel_row(i)
                K_ij = float(K_i[j])
                eta = 2.0 * K_ij - float(kernel_diag[i]) - float(kernel_diag[j])
                if eta >= 0.0:
                    continue

                # 6. Hitung dan clip alpha_j baru ke [L, H]
                aj_new = float(np.clip(aj_old - y[j] * (E_i - E_j) / eta, L, H))
                if abs(aj_new - aj_old) < 1e-5:
                    continue

                # 7. Hitung alpha_i baru agar syarat sum(alpha_k * y_k) = 0 tetap terpenuhi
                ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)

                d_ai = ai_new - ai_old
                d_aj = aj_new - aj_old

                # 8. Hitung kandidat bias b1, b2 lalu update b
                b1 = (b - E_i - y[i] * d_ai * float(kernel_diag[i]) - y[j] * d_aj * K_ij)
                b2 = (b - E_j - y[i] * d_ai * K_ij - y[j] * d_aj * float(kernel_diag[j]))

                if 0.0 < ai_new < C:
                    b = b1
                elif 0.0 < aj_new < C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                # 9. Simpan alpha baru & update vektor f_raw secara inkremental
                alpha[i] = ai_new
                alpha[j] = aj_new

                K_j = kernel_row(j)
                f_raw += (d_ai * y[i]) * K_i + (d_aj * y[j]) * K_j

                num_changed += 1

        if num_changed == 0:
            passes += 1
        else:
            passes = 0

    return alpha, float(b)
