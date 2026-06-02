# 🧠 CortiSoul — Analisis Kondisi Kesehatan Mental dari Teks Journaling

> **Capstone Project** · Analisis Data · Natural Language Processing (NLP)  
> Deteksi otomatis kondisi kesehatan mental berbasis teks journaling Bahasa Indonesia

---

## 📓 Lihat Notebook

Notebook utama berisi 131 cell dengan output visualisasi lengkap.
Karena ukurannya besar, GitHub tidak dapat merendernya secara langsung.
Gunakan salah satu link berikut:

| Platform | Link | Keterangan |
|---|---|---|
| **nbviewer** | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/capstone-project-CortiSoul-CC26-PSU353/Data-Science-CortiSoul/blob/main/Notebook_CortiSoul_Data_Science.ipynb) | Render lengkap dengan output |
| **Google Colab** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/capstone-project-CortiSoul-CC26-PSU353/Data-Science-CortiSoul/blob/main/Notebook_CortiSoul_Data_Science.ipynb) | Backup jika nbviewer tidak tersedia |

> ⚠️ Jika nbviewer menampilkan error 503, gunakan link Google Colab sebagai alternatif.

---

## 📌 Deskripsi Proyek

**CortiSoul** adalah sistem berbasis *Natural Language Processing* (NLP) yang dirancang untuk menganalisis teks journaling berbahasa Indonesia guna mendeteksi kondisi emosional dan tingkat stres pengguna secara otomatis.

Dalam lingkungan akademik dan profesional, banyak individu mengalami tekanan tinggi tanpa memiliki kesadaran yang cukup terhadap kondisi mental mereka. Proyek ini hadir sebagai solusi berbasis data untuk mendukung deteksi dini kondisi kesehatan mental melalui analisis teks.

---

## 🎯 Business Questions

1. **Token/frasa apa** yang paling merepresentasikan masing-masing dari 7 kategori kondisi mental, berdasarkan analisis **Word2Vec** dan **FastText** (top-10 kata paling mirip per kategori, `vector_size=100`, `window=5`) serta visualisasi word cloud?

2. **Bagaimana performa model NLP** menggunakan representasi vektor Word2Vec dan FastText dalam mengklasifikasikan 7 kondisi mental dengan akurasi minimal **80%** dan F1-score **≥ 0,75** melalui evaluasi Stratified 5-Fold Cross-Validation?

---

## 📦 Dataset

| Atribut | Detail |
|---|---|
| **Nama** | Mental Health Condition Classification Dataset |
| **Bahasa** | Indonesia (hasil terjemahan) |
| **Total data awal** | 103.488 entri |
| **Total data digunakan** | 9.163 entri |
| **Kolom** | `text` (teks journaling), `status` (label kondisi) |
| **Jumlah kelas** | 7 kondisi mental |

### Distribusi Kelas

| Kondisi Mental | Jumlah Data |
|---|---|
| Suicidal | 1.657 |
| Anxiety | 1.198 |
| Depression | 1.635 |
| Normal | 1.237 |
| Bipolar | 966 |
| Stress | 1.236 |
| Personality Disorder | 1234 |

---

## ⚙️ Pipeline Proyek

```
📥 Gathering Data
       ↓
🔍 Assessing Data     → Cek missing values, duplikat, distribusi kelas
       ↓
🧹 Cleaning Data      → Hapus duplikat, bersihkan URL/emoji/karakter khusus, normalisasi slang
       ↓
⚙️ Preprocessing      → Tokenisasi → Stopword Removal → Stemming (Sastrawi) → Filter token < 3 kata
       ↓
📊 EDA & Analisis Token → Word2Vec/FastText most_similar(), word cloud, heatmap overlap, token eksklusif
       ↓
🤖 Modeling           → Word2Vec + FastText × Random Forest + LinearSVC (4 kombinasi, Stratified 5-Fold CV)
       ↓
📈 Evaluasi           → Akurasi, F1-Score Macro, Confusion Matrix, Analisis Kesalahan
       ↓
🧪 A/B Testing        → Word2Vec vs FastText (Logistic Regression, McNemar Test)
```

---

## 🤖 Hasil Pemodelan

### Rekap Performa Semua Model (Stratified 5-Fold Cross-Validation)

| Representasi Vektor | Model Klasifikasi | Akurasi | F1 Macro |
|---|---|---|---|
| **Word2Vec** | **Random Forest** | **83.5% ± 0.7%** | **0.8598** |
| Word2Vec | LinearSVC | 82.0% | 0.8526 |
| FastText | LinearSVC | 82.3% | 0.8454 |
| FastText | Random Forest | 82.0% | 0.8459 |

### 🏆 Model Terbaik: Word2Vec + LinearSVC

- **Akurasi:** 82.9% ± 0.7% ✅ (target ≥ 80%)
- **F1 Macro:** 0.8511 ✅ (target ≥ 0.75)
- Word2Vec unggul karena representasi konteks kata berbasis token lebih stabil untuk teks journaling Bahasa Indonesia

### F1-Score per Kelas (Model Terbaik: Word2Vec + LinearSVC)

| Kondisi | F1-Score | Keterangan |
|---|---|---|
| Bipolar | 0.98 | ✅ Terbaik |
| Anxiety | 0.95 | ✅ |
| Stress | 0.97 | ✅ |
| Personality Disorder | 0.97 | ✅ |
| Normal | 0.76 | ✅ |
| Depression | 0.63 | ⚠️ Di bawah target |
| Suicidal | 0.66 | ⚠️ Di bawah target |

---

## 🧪 A/B Testing — Word2Vec vs FastText (Logistic Regression)

Pengujian murni pengaruh metode embedding dengan model yang identik (Logistic Regression, parameter dan seed sama):

| Metrik | Word2Vec (A) | FastText (B) | Selisih |
|---|---|---|---|
| Accuracy | 0.7995 | 0.7725 | -0.0270 |
| Precision | 0.8002 | 0.7730 | -0.0272 |
| Recall | 0.7995 | 0.7725 | -0.0270 |
| F1-Score | 0.7998 | 0.7725 | -0.0273 |

**Uji Statistik McNemar:** Statistik = 72.1287, p-value = **0.0000** → perbedaan **signifikan secara statistik** (p ≥ 0.05).
---

## 🔍 Analisis Token Eksklusif

Token eksklusif diidentifikasi via Word2Vec `most_similar()` — kata yang hanya muncul signifikan di satu kondisi tertentu:

| Kondisi | Jumlah Token Eksklusif | Contoh Token |
|---|---|---|
| Anxiety | 10 | *benak, lumpuh, kuasa, gerogot, ragu, tawan* |
| Bipolar | 10 | *mania, meluapluap, produktivitas, euforia, impulsif* |
| Normal | 10 | *meme, anime, santai, hobby, refreshing, seru* |
| Personality Disorder | 10 | *monster, ilusi, fluktuasi, minder, identitas* |
| Stress | 10 | *tenggat, rentet, wajib, deadline, burnout* |
| Depression | 7 | *bertahuntahun, takdir, obsesi, sengsara* |
| Suicidal | 8 | *pilih, alamiah, fantasi, kecut, acuh* |

**Anxiety** paling mudah dibedakan secara leksikal (10 token eksklusif). **Depression** dan **Suicidal** paling sulit — token eksklusif paling sedikit dan tumpang tindih semantik tinggi.

---

## 💡 Kesimpulan

**Pertanyaan 1 — Pola Token per Kondisi (Word2Vec & FastText):**
- Setiap kondisi memiliki token eksklusif yang menjadi ciri khasnya (diidentifikasi via Word2Vec `most_similar()`)
- **Anxiety** paling mudah dibedakan (10 token eksklusif: *benak, lumpuh, kuasa, gerogot, ragu, tawan*, dll.)
- **Stress** dicirikan kata *tenggat, rentet, wajib*; **Personality Disorder** oleh *monster, ilusi, fluktuasi, minder*
- **Depression** dan **Suicidal** memiliki hanya 6–8 token eksklusif — paling sulit dibedakan karena tumpang tindih semantik tertinggi antar kondisi
- Token umum (*pikir, orang, hidup*) muncul lintas kondisi dan tidak membedakan kelas secara spesifik

**Pertanyaan 2 — Performa Model:**
- Semua 4 kombinasi model berhasil melampaui target akurasi ≥ 80%
- Model terbaik **Word2Vec + LinearSVC** mencapai akurasi **82.9%** dan F1 Macro **0.8511**
- Depression dan Suicidal masih memiliki F1 < 0.75 dan sering tertukar satu sama lain (100 kasus suicidal → depression, 88 kasus depression → suicidal)

---

## 📋 Rekomendasi

1. **Tangani Class Imbalance** — Gunakan SMOTE atau augmentasi data khususnya untuk kelas *depression* dan *suicidal* yang memiliki F1 rendah
2. **Eksperimen Transformer** — Coba IndoBERT atau mBERT yang dioptimasi untuk Bahasa Indonesia
3. **Perluas Dataset** — Tambah data kelas minoritas (*personality disorder* = 730, *stress* = 1.046)
4. **Feature Engineering** — Tambahkan fitur sentimen, intensitas emosi, dan fitur pragmatik
5. **Strategi Anti-Overlap** — Kembangkan strategi khusus untuk membedakan pasangan *suicidal ↔ depression* dan *normal ↔ suicidal*
6. **Validasi Klinis** — Validasi model dengan psikolog atau profesional kesehatan mental sebelum deployment

---

## 🗂️ Struktur Folder

```
cortisoul/
│
├── 📁 dashboard/
│   ├── cortisoul_dashboard.py            # Aplikasi Streamlit
│   └── 📁 nb_images/                     # Gambar visualisasi dari notebook
│
├── 📁 data/
│   └── data_setelah_Preprocessing.csv    # Dataset hasil preprocessing
│
├── Notebook_CortiSoul_Data_Science.ipynb # Notebook analisis lengkap
├── README.md                             # Dokumentasi proyek
├── requirements.txt                      # Dependensi dashboard
└── url.txt                               # Link deployment
```

---

## 🚀 Cara Menjalankan Dashboard

### 1. Clone repositori

```bash
git clone https://github.com/Lindaputriani/Data-Science-CortiSoul.git
cd cortisoul
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan dashboard

```bash
cd dashboard
streamlit run cortisoul_dashboard.py
```

### 4. Buka browser

Dashboard akan otomatis terbuka di `http://localhost:8501`

> **Catatan:** Pastikan file `data_setelah_Preprocessing.csv` ada di folder `data/` agar grafik distribusi dan histogram panjang teks menampilkan data aktual (bukan estimasi fallback dari notebook).

---

## 🛠️ Tech Stack

| Kategori | Library |
|---|---|
| **Dashboard** | Streamlit |
| **Manipulasi Data** | Pandas, NumPy |
| **Visualisasi** | Plotly (express, graph_objects, subplots) |

---

## 📌 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
```

Install dependensi:

```bash
pip install -r requirements.txt
```

---

*CortiSoul Capstone Project · NLP untuk Kesehatan Mental · Bahasa Indonesia*