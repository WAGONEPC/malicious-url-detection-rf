<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-orange.svg?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg?logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</p>

<h1 align="center">Deteksi Malicious URL Menggunakan Algoritma Random Forest Classifier</h1>

<h2 align="center">PENELITI</h2>

<p align="center">
  Penelitian ini dilakukan secara mandiri sebagai bagian dari pemenuhan tugas<br>
  <strong>Keamanan Berbasis Sistem Cerdas (Semester 7)</strong> - Program Studi Teknik Informatika, Universitas Darma Persada.
</p>

<p align="center">
  <img 
    src="https://github.com/WAGONEPC.png"
    width="130"
    height="130"
    style="border-radius:50%;"
    alt="Sayyid Abdullah Azzam"
  />
</p>

<p align="center">
  <strong>Sayyid Abdullah Azzam</strong><br>
  <sub>Lead Researcher</sub><br>
  <a href="https://github.com/WAGONEPC">@WAGONEPC</a>
</p>

<h3 align="center">Peran Peneliti</h3>

<p align="center">
  <strong>Project Lead & Data Scraping</strong><br>
  <sub>Bertanggung jawab atas seluruh proses penelitian, mulai dari perancangan metodologi hingga pengolahan dan persiapan dataset.</sub>
</p>

<h4 align="center">Kontribusi</h4>

<div align="center">

| No. | Kontribusi |
|:---:|:---|
| 1 | Perancangan metodologi penelitian |
| 2 | Pengumpulan dataset dari platform Kaggle |
| 3 | Data cleaning dan preprocessing |
| 4 | Persiapan dataset untuk proses analisis dan klasifikasi |

</div>

<br>

<p align="center">
  <em>Repository ini disusun untuk memenuhi tugas <strong>Ujian Tengah Semester (UTS)</strong> mata kuliah <strong>Keamanan Berbasis Sistem Cerdas (Semester 7)</strong> pada Program Studi Teknik Informatika, <strong>Universitas Darma Persada (UNSADA)</strong>.</em>
</p>

---

## Gambaran Umum Proyek

Pertumbuhan kejahatan siber berbasis rekayasa sosial (*phishing*), penyusupan perangkat perusak (*malware*), serta perusakan situs (*defacement*) kian mengancam keamanan data pengguna internet. Salah satu vektor serangan utama adalah distribusi tautan web berbahaya (*malicious URLs*).

Proyek ini mengimplementasikan sistem deteksi URL berbahaya otomatis berbasis **Machine Learning** menggunakan algoritma **Random Forest Classifier**. Sistem ini mengekstrak **8 fitur leksikal numerik** dari struktur teks URL, mengatasi permasalahan ketimpangan kelas (*class imbalance*) menggunakan teknik **SMOTE**, menyimpan model terlatih ke disk lokal, serta menyediakan fungsi inferensi interaktif `scan_url()` yang siap digunakan secara *real-time* untuk menentukan apakah suatu URL berstatus **AMAN** atau **BLOKIR**.

Untuk menjaga modularitas dan keterbacaan kode, alur implementasi dipecah menjadi **dua Jupyter Notebook**:
1. **Bagian 1 ([`malicious_url_detection.ipynb`](malicious_url_detection.ipynb))**: Tahap 1 s.d. 3 (Akuisisi Data, EDA, Data Preprocessing & Feature Engineering).
2. **Bagian 2 ([`model_training_and_inference.ipynb`](model_training_and_inference.ipynb))**: Tahap 4 s.d. 8 (Data Splitting, SMOTE, Training Random Forest, Evaluasi Testing, & Real-Time Inference).

---

## Diagram Alur Sistem (Pipeline Architecture)

Berikut adalah visualisasi alur komprehensif mulai dari akuisisi dataset hingga pengujian URL:

![Alur Deteksi URL Berbahaya](Alur%20Deteksi%20URL%20Berbahaya.png)

```mermaid
flowchart TD
    A["1. Akuisisi Data (Kaggle / Lokal)<br/>malicious_phish.csv (20.000 sampel terukur)"] --> B["2. Exploratory Data Analysis (EDA)<br/>Distribusi Kategori URL (Bar Chart) & Identifikasi Imbalance"]
    B --> C["3. Data Preprocessing & Feature Engineering<br/>• Cleaning: dropna & hapus duplikat<br/>• Binary Label: Benign (0) vs Bahaya (1)<br/>• Ekstraksi 8 Fitur Leksikal URL"]
    C -->|Simpan dataset/extracted_features.csv| D["4. Data Splitting (80% Training : 20% Testing)<br/>train_test_split(stratify=y) - Wajib sebelum SMOTE"]
    D -->|80% Data Latih| E["5. Penanganan Imbalance (SMOTE)<br/>Oversampling kelas minoritas hingga rasio 50:50"]
    D -->|20% Data Uji (Tetap Murni/Asli)| H["7. Evaluasi Model (Testing)<br/>Uji model terhadap 20% Data Testing<br/>Confusion Matrix, Accuracy, Precision, Recall, F1-Score"]
    E --> F["6. Training Model<br/>Random Forest (100 Trees)<br/>Fit ke X_train_smote & y_train_smote"]
    F -->|Simpan Model| G["Model Terlatih<br/>(model/random_forest_model.joblib)"]
    G --> H
    G --> I["8. Inference / Deployment (Uji URL Baru)<br/>Input URL Baru ➔ Ekstrak 8 Fitur ➔ Prediksi RF ➔ AMAN / BLOKIR"]
```

---

## Urutan Penjelasan Per Tahap (Alur Koding)

### 1. Akuisisi Data (Kaggle & Lokal)
Dataset yang digunakan adalah **Malicious URLs dataset** (`malicious_phish.csv`) karya Sidharth Sharma yang tersedia di platform [Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset). Dataset ini memuat 651.191 URL dengan klasifikasi `benign`, `defacement`, `phishing`, dan `malware`.

Dataset dapat diunduh langsung menggunakan pustaka resmi `kagglehub` atau dimuat secara instan dari folder lokal [`dataset/malicious_phish.csv`](dataset/malicious_phish.csv):

```python
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "malicious_phish.csv"
local_csv = os.path.join("dataset", file_path)

if os.path.exists(local_csv):
    df_raw = pd.read_csv(local_csv)
else:
    df_raw = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "sid321axn/malicious-urls-dataset",
        file_path,
    )

print("First 5 records:", df_raw.head())
```

> **Catatan Pengambilan Sampel Terukur (20.000 Baris)**:  
> Untuk menjamin proses pelatihan, evaluasi, dan demonstrasi video berjalan sangat cepat dan responsif (hanya membutuhkan hitungan detik), diambil sampel representatif sebanyak **20.000 baris** menggunakan `df.sample(n=20000, random_state=42)`. Parameter ini dapat diubah dengan mudah jika pengguna ingin melatih pada keseluruhan dataset asli (651.191 baris).

---

### 2. Exploratory Data Analysis (EDA)
Pada tahap ini, dilakukan analisis distribusi data awal untuk:
- Mengecek kelengkapan data (*missing values* / null) serta struktur tipe data.
- Menampilkan visualisasi diagram batang (*countplot*) dan *pie chart* proporsi kelas.
- **Identifikasi Ketidakseimbangan Data (*Class Imbalance*)**:  
  Hasil EDA menunjukkan bahwa jumlah URL Aman (*benign*) jauh lebih banyak dibandingkan URL kategori berbahaya (`defacement`, `phishing`, `malware`). Ketimpangan kelas ini menuntut penggunaan teknik penyeimbangan data sintetis (**SMOTE**) pada tahap pelatihan agar model tidak mengalami bias atau berat sebelah ke kelas mayoritas.

---

### 3. Data Preprocessing & Feature Engineering
Tahap persiapan data terdiri dari:
1. **Pembersihan Data**: Menghapus nilai kosong (*null*) menggunakan `dropna()` dan menghapus baris URL duplikat.
2. **Penyederhanaan Label (Biner)**:
   - **`0`**: **Aman (Benign)**
   - **`1`**: **Bahaya (Malicious)** (menggabungkan *phishing*, *defacement*, dan *malware*).
3. **Ekstraksi 8 Fitur Leksikal URL**:

| No | Nama Fitur | Nama Kolom | Deskripsi Teknis | Signifikansi Keamanan Siber |
|:---:|:---|:---|:---|:---|
| 1 | `panjang_url` | `url_length` | Panjang total karakter URL | URL serangan sering dipanjangkan untuk menyamarkan subdomain atau menyelipkan *tracking token*. |
| 2 | `jml_titik` | `count_dot` | Jumlah tanda titik (`.`) | Menandakan struktur subdomain bertingkat (*deep subdomains*) yang kerap dipakai memalsukan brand resmi. |
| 3 | `jml_strip` | `count_hyphen` | Jumlah tanda hubung (`-`) | Sering dimanfaatkan pada domain tipuan (*typosquatting*) seperti `unsada-portal-login.com`. |
| 4 | `jml_slash` | `count_slash` | Jumlah garis miring (`/`) | Mengindikasikan kedalaman direktori path tempat file berbahaya diletakkan. |
| 5 | `simbol_at` | `count_at` | Jumlah simbol `@` | Simbol `@` membuat browser mengabaikan bagian sebelum simbol sebagai kredensial, trik klasik pengalihan URL. |
| 6 | `jml_angka` | `count_digits` | Jumlah karakter angka (0-9) | Domain berbahaya sering memuat deretan angka acak atau hash *session id* palsu. |
| 7 | `penggunaan_ip` | `has_ip` | Deteksi pola IPv4 langsung | Bernilai `1` jika URL menggunakan IP publik langsung tanpa nama domain terdaftar, `0` jika tidak. |
| 8 | `kata_kunci_phish` | `has_phish_keywords` | Deteksi kata kunci mencurigakan | Bernilai `1` jika URL memuat kata kunci penipuan (*login, verify, update, banking, secure, account, signin, password, webscr, wallet, admin*), `0` jika tidak. |

Dataset hasil preprocessing dan ekstraksi fitur disimpan ke [`dataset/extracted_features.csv`](dataset/extracted_features.csv) untuk digunakan langsung pada Bagian 2.

---

### 4. Data Splitting (80% Training : 20% Testing)
> **Prinsip Utama: Mencegah Data Leakage (Kebocoran Data)**  
> Pembagian dataset (*train-test split*) **wajib dilakukan sebelum** algoritma SMOTE diterapkan.  
> Jika SMOTE diterapkan sebelum pemisahan data, sampel sintetis yang dibuat berdasarkan data uji akan bocor ke data latih, menyebabkan performa model tampak tinggi secara semu (*overoptimistic*) dan gagal digeneralisasi di dunia nyata.

**Alur yang diterapkan:**
1. Dataset dipisah menjadi **80% Data Training** dan **20% Data Testing** secara terstratifikasi (`stratify=y`).
2. Data Testing (**20%**) dibiarkan **tetap murni/asli** tanpa SMOTE agar menjadi tolok ukur pengujian yang objektif.

---

### 5. Penanganan Imbalance (SMOTE)
Algoritma **SMOTE** (*Synthetic Minority Over-sampling Technique*) diterapkan **hanya pada 80% Data Training** (`X_train` dan `y_train`). SMOTE membentuk data sintetis pada kelas minoritas (URL Bahaya) hingga perbandingan antara kelas Aman dan Bahaya seimbang sempurna **50:50 (1:1)**.

---

### 6. Training Model Random Forest & Penyimpanan Model
Model yang digunakan adalah **Random Forest Classifier** dengan konfigurasi:
- `n_estimators=100` (100 pohon keputusan / *decision trees*)
- `random_state=42` (menjamin replikasi hasil yang konsisten)
- `n_jobs=-1` (pemanfaatan seluruh core prosesor untuk pelatihan kilat)

Model dilatih menggunakan data training yang sudah diseimbangkan (`X_train_smote` dan `y_train_smote`), dianalisis tingkat kepentingan fiturnya (*Feature Importance*), kemudian disimpan ke disk lokal:
```python
import joblib
joblib.dump(rf_model, "model/random_forest_model.joblib")
```
Model terlatih tersimpan di folder [`model/random_forest_model.joblib`](model/random_forest_model.joblib) sehingga siap di-load sewaktu-waktu tanpa perlu training ulang.

---

### 7. Evaluasi Model (Testing)
Pengujian dilakukan pada 20% Data Testing murni (`X_test`). Metrik yang dianalisis mencakup:
- **Confusion Matrix**: Menghitung True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN).
- **Accuracy**: Mengukur persentase kebenaran tebakan secara agregat (~85%+).
- **Precision**: Mengukur seberapa akurat saat model menyatakan suatu tautan berbahaya.
- **Recall (Fokus Utama Keamanan Siber)**:  
  > *Dalam konteks pertahanan siber, metrik **Recall untuk kelas Bahaya (1)** adalah parameter paling penting*. Recall yang tinggi memastikan seminimal mungkin link phishing/malware yang lolos (*False Negative* sekecil mungkin), sehingga melindungi pengguna akhir dari pembajakan akun dan kebocoran data sensitif.
- **F1-Score**: Rata-rata harmonis antara Precision dan Recall.

---

### 8. Inference / Deployment (Uji URL Baru: Aman / Blokir)
Dibangun fungsi inferensi siap pakai `scan_url(url)` yang memuat model tersimpan dari `model/random_forest_model.joblib`. Fungsi ini secara instan:
1. Menerima string URL baru dari pengguna.
2. Mengekstrak 8 fitur leksikal numerik secara otomatis.
3. Melakukan inferensi menggunakan model Random Forest.
4. Menghasilkan keputusan: **AMAN** (Tautan Terverifikasi) atau **BLOKIR** (Tautan Berbahaya) beserta persentase keyakinan model (*Confidence Score*).

**Hasil Pengujian Skenario Riil:**
- **URL Resmi Kampus UNSADA**:
  - `unsada.ac.id/tentang-kampus/sejarah` ➔ **AMAN** (Keyakinan: ~98.5%)
  - `portal.unsada.ac.id/berita.html` ➔ **AMAN** (Keyakinan: ~88.3%)
- **URL Populer Terverifikasi**:
  - `google.com/search?q=cybersecurity` ➔ **AMAN** (Keyakinan: ~100.0%)
- **URL Serangan Phishing Berbasis IP**:
  - `http://192.168.1.105/login-unsada-verify-account.php` ➔ **BLOKIR**
- **URL Serangan Phishing Finansial**:
  - `http://paypal-security-update.com/signin/verify.php` ➔ **BLOKIR**
- **URL Eksploitasi / Defacement**:
  - `http://www.garage-pirenne.be/index.php?option=com_content...` ➔ **BLOKIR**

---

## Cara Menjalankan Proyek

### Opsi 1: Menjalankan di Komputer Lokal (VS Code / Jupyter Notebook)
1. Clone repositori ini:
   ```bash
   git clone https://github.com/WAGONEPC/malicious-url-detection-rf.git
   cd malicious-url-detection-rf
   ```
2. Pasang seluruh dependensi melalui file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan notebook secara berurutan:
   - **Langkah 1**: Buka dan jalankan [`malicious_url_detection.ipynb`](malicious_url_detection.ipynb) (Tahap 1 - 3). Ini akan mengekstrak fitur dan membuat file `dataset/extracted_features.csv`.
   - **Langkah 2**: Buka dan jalankan [`model_training_and_inference.ipynb`](model_training_and_inference.ipynb) (Tahap 4 - 8). Ini akan melatih model, menyimpan model ke `model/random_forest_model.joblib`, melakukan evaluasi, dan menguji fungsi `scan_url()`.

### Opsi 2: Menjalankan di Google Colab
1. Buka [Google Colab](https://colab.research.google.com/).
2. Unggah file [`malicious_url_detection.ipynb`](malicious_url_detection.ipynb) dan jalankan sel-selnya.
3. Unggah file [`model_training_and_inference.ipynb`](model_training_and_inference.ipynb) untuk melanjutkan pemodelan dan pengujian inferensi.

---

## Struktur Direktori Repositori

```text
malicious-url-detection-rf/
├── Alur Deteksi URL Berbahaya.png    # Infografis & diagram alur lengkap sistem deteksi
├── LICENSE                            # Lisensi sumber terbuka MIT
├── README.md                          # Dokumentasi resmi proyek UTS
├── requirements.txt                   # Daftar dependensi modul Python
│
├── dataset/                           # Direktori penyimpanan data lokal
│   ├── malicious_phish.csv            # Dataset mentah dari Kaggle (651.191 baris)
│   └── extracted_features.csv         # Dataset hasil ekstraksi 8 fitur leksikal
│
├── model/                             # Direktori penyimpanan model machine learning
│   └── random_forest_model.joblib     # Model Random Forest Classifier terlatih
│
├── malicious_url_detection.ipynb      # Notebook Bagian 1: Tahap 1 - 3 (Akuisisi, EDA, Feature Engineering)
└── model_training_and_inference.ipynb # Notebook Bagian 2: Tahap 4 - 8 (Splitting, SMOTE, Training, Evaluasi, Inference)
```

---

## Identitas Mahasiswa

- **Nama**: Sayyid Abdullah Azzam
- **Program Studi**: Teknik Informatika (Semester 7)
- **Mata Kuliah**: Keamanan Berbasis Sistem Cerdas
- **Institusi**: Universitas Darma Persada (UNSADA), Jakarta

---

## Lisensi

Proyek ini dirilis di bawah lisensi terbuka [MIT License](LICENSE). Bebas digunakan dan dikembangkan untuk keperluan akademik dan penelitian keamanan siber.
