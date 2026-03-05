# 🎭 Klasifikasi Emosi Bahasa Indonesia

**NoLimit Indonesia --- Data Scientist Hiring Test**\
Opsi yang Dipilih: **Classification (Emotion Classification)**

------------------------------------------------------------------------

## 📌 Tujuan Proyek

Membangun solusi Natural Language Processing (NLP) untuk melakukan
**klasifikasi emosi pada ulasan pelanggan berbahasa Indonesia**
menggunakan model Hugging Face dan pendekatan embedding dengan workflow
yang bersih, terstruktur, dan reproducible.

------------------------------------------------------------------------

## 📂 Dataset

### 📛 Nama Dataset

**PRDECT-ID: Indonesian Emotion Classification**

### 📝 Deskripsi

PRDECT-ID merupakan dataset ulasan produk berbahasa Indonesia yang telah
dianotasi dengan label emosi dan sentimen. Dataset ini dikumpulkan dari
salah satu e-commerce terbesar di Indonesia, yaitu **Tokopedia**.

Dataset berisi ulasan dari **29 kategori produk**, dengan setiap ulasan
diberi satu label emosi berikut:

-   Love
-   Happiness
-   Anger
-   Fear
-   Sadness

Proses anotasi dilakukan oleh sekelompok annotator berdasarkan kriteria
anotasi emosi yang disusun oleh ahli psikologi klinis.

Selain teks ulasan, dataset juga menyertakan atribut tambahan seperti: -
Location - Price - Overall Rating - Number Sold - Total Review -
Customer Rating

Namun pada proyek ini, hanya kolom teks dan label emosi yang digunakan.

### 📊 Ringkasan Dataset

-   Total Data: 5400
-   Jumlah Kelas: 5
-   Bahasa: Indonesia

### 🔗 Sumber Dataset

Rhio Sutoyo\
(https://www.kaggle.com/datasets/jocelyndumlao/prdect-id-indonesian-emotion-classification/data)

------------------------------------------------------------------------

## 🧠 Pendekatan dan Model

### 1️⃣ Baseline --- TF-IDF + Logistic Regression

-   TF-IDF (Unigram + Bigram)
-   Logistic Regression (class-balanced)

### 2️⃣ Embedding Model --- SentenceTransformer + LinearSVC

-   Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
-   Dense embedding 384 dimensi
-   Classifier: LinearSVC

### 3️⃣ Fine-Tuned Transformer --- IndoBERT

-   Model: `indobenchmark/indobert-base-p1`
-   Fine-tuning: 4 Epoch
-   Max sequence length: 128
-   Learning rate: 2e-5
-   Batch size: 16 (Train & Eval)
-   Optimizer: AdamW

------------------------------------------------------------------------

## 📊 Hasil Evaluasi

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 0.598 | 0.566 |
| SentenceTransformer + LinearSVC | 0.606 | 0.565 |
| Fine-Tuned IndoBERT | 0.70+ | 0.67+ |

Model **Fine-Tuned IndoBERT** memberikan performa terbaik karena mampu memahami konteks bahasa secara lebih mendalam dibanding metode klasik.

------------------------------------------------------------------------

## 🔄 Alur Pipeline End-to-End

1.  Data Loading
2.  Data Cleaning
3.  Label Encoding
4.  Train-Test Split (80:20)
5.  Baseline Modeling (TF-IDF)
6.  Embedding Modeling (SentenceTransformer)
7.  Fine-Tuning IndoBERT
8.  Evaluasi (Accuracy & Macro F1)
9.  Analisis Confusion Matrix
10. Contoh Inferensi

Flowchart tersedia pada file `flowchart.png`.

------------------------------------------------------------------------

## ▶️ Cara Menjalankan

### 1️⃣ Clone Repository

``` bash
git clone <url-repository-anda>
cd nolimit-ds-test-<nama>
```

### 2️⃣ Install Dependency

``` bash
pip install -r requirements.txt
```

### 3️⃣ Jalankan Notebook

Buka file `.ipynb` menggunakan: - Jupyter Notebook - VSCode - Google
Colab

------------------------------------------------------------------------

## 🧪 Contoh Inferensi

``` python
predict_bert("Saya sangat kecewa dengan pelayanan ini")
```

Output:

    Sadness

------------------------------------------------------------------------

## 📁 Struktur Repository

    nolimit-ds-test-<nama>/
    │
    ├── notebook.ipynb
    ├── requirements.txt
    ├── README.md
    ├── flowchart.png
    └── sample_data.csv
