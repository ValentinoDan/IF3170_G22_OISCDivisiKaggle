# IF3170_G22_OISCDivisiKaggle - Tugas Besar 2 Inteligensi Artifisial

Repository pengerjaan Tugas Besar 2 IF3170 Inteligensi Artifisial (Supervised Learning) oleh Kelompok G22 - OISCDivisiKaggle. Tugas besar ini bertujuan untuk menyelesaikan kasus prediksi pembatalan pemesanan penginapan (*hotel booking cancellation*, dengan target `is_canceled`) menggunakan lima algoritma *supervised learning* yang diimplementasikan dari **scratch** dan dibandingkan dengan pustaka `scikit-learn`:

1. **K-Nearest Neighbors (KNN)**
2. **Decision Tree Learning (CART)**
3. **Logistic Regression** 
4. **Support Vector Machine (SVM)** 
5. **Bayesian Network (Naive Bayes)** 

---

## Struktur Direktori

```text
IF3170_G22_OISCDivisiKaggle/
├── dataset/                    # Dataset kompetisi Kaggle (train.csv, test.csv, sample_submission.csv)
├── docs/                       # Dokumentasi dan laporan Tugas Besar 2
├── output/                     # Hasil eksekusi program (submission.csv, visualisasi graf/kurva, dan model .pkl)
│   └── models/                 # Berkas model terlatih yang disimpan (.pkl)
├── src/                        # Kode sumber implementasi algoritma from scratch & notebook eksperimen
│   ├── decision_tree/          # Implementasi CART Decision Tree, Optimized Tree, dan visualisasi pohon
│   ├── knn/                    # Implementasi K-Nearest Neighbors (KNN)
│   ├── logistic_regression/    # Implementasi Logistic Regression (L1, L2, Elastic Net) & LROptimized
│   ├── naive_bayes/            # Implementasi modular Naive Bayes, Feature-Weighted Optimizer, & Visualizer
│   ├── svm/                    # Implementasi Support Vector Machine (QP & SMO solver)
│   ├── preprocessing.py        # Modul utilitas pembersihan dan prapemrosesan data
│   ├── utils.py                # Modul utilitas penyimpanan model, evaluasi, dan visualisasi
│   └── notebook.ipynb          # Notebook utama alur CRISP-DM, evaluasi, tuning, & pembuatan submisi Kaggle
├── requirements.txt            # Daftar dependensi pustaka Python
└── README.md                   # Deskripsi repository, panduan instalasi/eksekusi, dan pembagian tugas
```

---

## Cara Setup dan Run Program

### 1. Pre-Req
Pastikan sistem sudah terpasang **Python 3.10+** dan `pip`.

### 2. Clone Repository
```bash
git clone https://github.com/ValentinoDan/IF3170_G22_OISCDivisiKaggle.git
cd IF3170_G22_OISCDivisiKaggle
```

### 3. Buat dan Mengaktifkan Virtual Environment (Recommended)

- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

### 4. Instalasi Dependensi
Instal seluruh pustaka yang dibutuhkan dari `requirements.txt` (serta `qpsolvers` dan `osqp` untuk solver QP pada SVM):
```bash
pip install -r requirements.txt qpsolvers osqp
```

### 5. Menjalankan Program (`src/notebook.ipynb`)
Seluruh pipeline eksperimen (mulai dari *Data Understanding*, *Data Cleaning & Preprocessing*, *Modeling & Validation*, *Hyperparameter Tuning*, *Perbandingan dengan scikit-learn*, *Save/Load Model*, *Kaggle Submission*, *Error Analysis*, hingga *Cost-Sensitive Classification*) dijalankan melalui notebook utama [`src/notebook.ipynb`](src/notebook.ipynb):

- **Menggunakan Jupyter Notebook / JupyterLab:**
  ```bash
  jupyter notebook src/notebook.ipynb
  # atau
  jupyter lab src/notebook.ipynb
  ```
- **Menggunakan VS Code:**
  1. Buka berkas `src/notebook.ipynb` di VS Code.
  2. Pilih kernel Python dari virtual environment `.venv` yang telah dibuat.
  3. Klik **Run All** untuk mengeksekusi seluruh *cell* secara berurutan.
  4. Hasil prediksi akhir akan tersimpan otomatis di `output/submission.csv` dan seluruh model terlatih akan tersimpan di `output/models/`.

---

## Pembagian Tugas Anggota Kelompok

| NIM | Nama | Pembagian Tugas |
| :---: | :--- | :--- |
| **13524041** | Nathan Adhika Santosa | Implementasi Algoritma Decision Tree (CART), Optimized Decision Tree, Visualisasi Percabangan DTL, dan Penulisan Laporan |
| **13524080** | Renuno Yuqa Frinardi | Implementasi Algoritma Bayesian Network (Naive Bayes), Optimized Feature-Weighted NB, Visualisasi Graf BN, Cost-Sensitive Classification, Notebook, dan Penulisan Laporan |
| **13524092** | Timothy Bernard Soeharto | Implementasi Algoritma K-Nearest Neighbors (KNN), Optimasi KNN, dan Penulisan Laporan |
| **13524104** | Valentino Daniel Kusumo | Implementasi Algoritma Logistic Regression (L1, L2, Elastic Net), Optimized LR (Adam), Integrasi Notebook, dan Penulisan Laporan |
| **13524146** | Leonardus Brain Fatolosja | Implementasi Algoritma Support Vector Machine (Linear, Polynomial, Gaussian), Optimasi SMO Solver, Integrasi Notebook, dan Penulisan Laporan |
