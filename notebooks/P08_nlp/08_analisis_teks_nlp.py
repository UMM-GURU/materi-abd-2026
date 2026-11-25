# ---
# Pertemuan 8: Analisis Teks (NLP) pada Data Besar
# Kuliah Analisis Big Data - Semester Ganjil 2026
# Dosen Pengampu: Yuda Munarko
# ---
#
# Notebook ini adalah panduan hands-on lab untuk Pertemuan 8.
# Silakan jalankan cell di bawah ini secara berurutan.

# %% [markdown]
# # Pertemuan 8: Analisis Teks (NLP) pada Data Besar
#
# ## Capaian Pembelajaran Pertemuan (Sub-CPMK 3 - Kognitif K4, K5):
# 1. Memahami karakteristik data teks tak terstruktur dan tantangan pemrosesan teks bahasa Indonesia berskala besar (*high dimensionality, vocabulary explosion, slang, code-mixing*).
# 2. Mengimplementasikan pipeline pembersihan teks (*preprocessing*) performa tinggi memanfaatkan **Polars String Expressions** berbasis Rust tanpa hambatan *Python GIL*.
# 3. Membandingkan efisiensi waktu komputasi pembersihan teks antara *Polars String Expressions* dan fungsi Python konvensional (`apply(lambda)`).
# 4. Menerapkan kamus normalisasi *slang* marketplace dan penyaringan *stopwords* terarah yang mempertahankan kata negasi (*"tidak", "bukan", "kurang"*).
# 5. Mengekstrak representasi vektor teks menggunakan **TF-IDF Sparse Matrix** dengan pembatasan kosa kata (`max_features`) untuk mencegah *Out-of-Memory (OOM)*.
# 6. Membangun dan mengevaluasi model klasifikasi sentimen ulasan produk menggunakan **Scikit-learn Pipeline** (*Logistic Regression* dan *Multinomial Naive Bayes*).
# 7. Memahami integrasi model *pre-trained transformer* bahasa Indonesia (**IndoNLU / IndoBERT**) dari ekosistem **Hugging Face**.
# 8. Mempersiapkan checkpoint pengumpulan **Tugas 1 Milestone 3** serta pembagian kerja tim **Tugas 2 Phase 2**.

# %% [markdown]
# ---
# ## Bagian 1: Inisialisasi Lingkungan & Import Library
#
# Pada tahap ini, kita memuat pustaka pengolah data tabular performa tinggi (*Polars, DuckDB*), visualisasi (*Plotly*), dan *Machine Learning* (*Scikit-learn*).

# %%
import time
import os
import re
import numpy as np
import polars as pl
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print(f"Versi Polars    : {pl.__version__}")
print(f"Versi DuckDB    : {duckdb.__version__}")
print("Lingkungan NLP Analisis Big Data siap digunakan!")

# %% [markdown]
# ---
# ## Bagian 2: Pembangkitan Dataset Sintetis Ulasan E-Commerce Indonesia (~100.000 Baris)
#
# Di dunia analitik nyata, ulasan marketplace (*Tokopedia, Shopee, Blibli*) berjumlah ratusan ribu baris dengan variasi sentimen, kata gaul, singkatan, dan tanda baca yang tidak seragam.
#
# Kita membangkitkan 100.000 data ulasan sintetis dan menyimpannya ke dalam file **Apache Parquet** terkompresi di folder `data/processed/nlp_reviews/`.

# %%
output_dir = "data/processed/nlp_reviews"
os.makedirs(output_dir, exist_ok=True)
filepath_parquet = os.path.join(output_dir, "ulasan_ecommerce_indonesia.parquet")

jumlah_ulasan = 100_000
np.random.seed(2026)

if not os.path.exists(filepath_parquet):
    print(f"Membangkitkan {jumlah_ulasan:,} data ulasan e-commerce sintetis...")
    
    # Template teks ulasan positif, negatif, dan netral bahasa Indonesia
    template_positif = [
        "Barang bagus bgt sesuai deskripsi pengiriman super cepat kurir ramah!",
        "Kualitas produk sangat mantap original seller fast respon recommended bgt",
        "Packing aman bubble wrap tebal barang mendarat selamat makasih gan",
        "Harga murah tapi kualitas bukan kaleng kaleng sangat puas belanja disini",
        "Sesuai pesanan berfungsi dgn baik semoga awet bintang lima berbicara",
        "Pengiriman kilat baru pesan kemarin sore hari ini sdh sampai top bgt",
        "Bagus pol warna cakep pas bgt ukurannya next pasti repeat order disini!",
        "Produk oke banget gak ada cacat sedikitpun respon penjual gercep jempolan",
        "Pelayanan terbaik barang ori segel resmi jangan ragu beli di toko ini",
        "Suka banget sama bahannya tebel dan jahitan rapih terima kasih ya"
    ]
    
    template_negatif = [
        "Kecewa bgt barang rusak pas sampai seller tidak tanggung jawab sama sekali",
        "Pengiriman lama bgt seminggu baru nyampe barang tidak sesuai ekspektasi",
        "Kualitas jelek mudah patah sangat menyesal beli disini jangan mau tertipu",
        "Barang tidak ori alias palsu kemasan robek parah nyesel bgt belanja disini",
        "Seller slow respon ditanya tidak dibalas pengajuan komplain berbelit belit",
        "Ukurannya kekecilan tidak sesuai tabel ukuran bahan tipis menerawang",
        "Cacat pabrik tidak bisa nyala rugi beli barang beginian buang buang uang",
        "Warna yg dikirim salah minta hitam dikirim merah pelayanan buruk",
        "Packing sangat buruk cuma dibungkus plastik tipis tanpa bubble wrap pecah",
        "Kapok beli disini pelayanannya mengecewakan barang tidak berfungsi sama sekali"
    ]
    
    template_netral = [
        "Barang sudah sampai biasa saja sesuai dengan harganya standar lah",
        "Packing standar pengiriman lumayan cepat belum dicoba semoga berfungsi",
        "Kualitas lumayan bagus untuk harga segini tidak ada yg spesial",
        "Produk sesuai deskripsi pengiriman standar tidak cepat tidak lambat",
        "Barang diterima dalam kondisi baik belum sempat dites pemakaian",
        "Biasa aja bahannya agak tipis tapi masih wajar untuk harga segitu"
    ]
    
    kategori_produk = ["Elektronik", "Fashion Pria", "Fashion Wanita", "Kebutuhan Rumah", "Kecantikan & Perawatan", "Gadget & Aksesoris"]
    marketplace_platform = ["Tokopedia", "Shopee", "Bukalapak", "Lazada", "Blibli"]
    
    # Distribusi sentimen riil (imbalanced: 70% positif, 20% negatif, 10% netral)
    pilihan_sentimen = np.random.choice(["positif", "negatif", "netral"], size=jumlah_ulasan, p=[0.70, 0.20, 0.10])
    
    ulasan_list = []
    rating_list = []
    
    for sent in pilihan_sentimen:
        if sent == "positif":
            base_teks = np.random.choice(template_positif)
            rating = np.random.choice([4, 5], p=[0.30, 0.70])
        elif sent == "negatif":
            base_teks = np.random.choice(template_negatif)
            rating = np.random.choice([1, 2], p=[0.75, 0.25])
        else:
            base_teks = np.random.choice(template_netral)
            rating = 3
            
        # Menambahkan noise (tanda seru berulang, kapitalisasi acak, typo kecil)
        if np.random.rand() > 0.6:
            base_teks = base_teks.upper() if np.random.rand() > 0.8 else base_teks + " !!! 👍🙏"
            
        ulasan_list.append(base_teks)
        rating_list.append(rating)
        
    df_raw = pl.DataFrame({
        "review_id": [f"REV-{i:07d}" for i in range(1, jumlah_ulasan + 1)],
        "platform": np.random.choice(marketplace_platform, size=jumlah_ulasan),
        "kategori": np.random.choice(kategori_produk, size=jumlah_ulasan),
        "teks_ulasan": ulasan_list,
        "rating": rating_list,
        "sentimen_label": pilihan_sentimen
    })
    
    df_raw.write_parquet(filepath_parquet, compression="zstd")
    print(f"Dataset berhasil disimpan: {filepath_parquet} ({df_raw.shape[0]:,} baris)")
else:
    print(f"Dataset Parquet sudah tersedia di: {filepath_parquet}")

# %% [markdown]
# ---
# ## Bagian 3: Eksplorasi Data Teks Menggunakan DuckDB & Polars
#
# Kita menganalisis distribusi data ulasan, panjang karakter, dan sebaran sentimen secara cepat menggunakan **DuckDB in-process**.

# %%
con = duckdb.connect()

# Query ringkasan distribusi ulasan per platform dan sentimen
query_ringkasan = f"""
    SELECT
        platform,
        sentimen_label,
        COUNT(*) AS total_ulasan,
        ROUND(AVG(rating), 2) AS rata_rata_rating,
        ROUND(AVG(LENGTH(teks_ulasan)), 1) AS rata_panjang_karakter
    FROM '{filepath_parquet}'
    GROUP BY platform, sentimen_label
    ORDER BY platform, sentimen_label;
"""

# Menggunakan con.sql(...).pl() untuk konversi ke Polars DataFrame dari DuckDBPyRelation
try:
    df_ringkasan = con.sql(query_ringkasan).pl()
except (AttributeError, Exception):
    df_ringkasan = pl.from_arrow(con.execute(query_ringkasan).arrow())

print("Ringkasan Distribusi Ulasan via DuckDB:")
display(df_ringkasan)

# %% [markdown]
# ### 3.1 Visualisasi Distribusi Rating & Sentimen dengan Plotly
# Kita melihat visualisasi sebaran data ulasan untuk memahami ketimpangan kelas (*class imbalance*).

# %%
df_sentimen_count = (
    df_ringkasan
    .group_by("sentimen_label")
    .agg(pl.col("total_ulasan").sum().alias("jumlah"))
    .to_pandas()
)

fig_bar = px.bar(
    df_sentimen_count,
    x="sentimen_label",
    y="jumlah",
    color="sentimen_label",
    color_discrete_map={"positif": "#10b981", "netral": "#f59e0b", "negatif": "#f43f5e"},
    title=f"Distribusi Kelas Sentimen pada {jumlah_ulasan:,} Ulasan E-Commerce",
    labels={"sentimen_label": "Kategori Sentimen", "jumlah": "Jumlah Ulasan"},
    text_auto=True
)
fig_bar.update_layout(height=400, template="plotly_white")
fig_bar.show()

# %% [markdown]
# ---
# ## Bagian 4: Benchmark Pembersihan Teks: Polars Expressions vs Python `apply(lambda)`
#
# Mengapa kita tidak boleh menggunakan `apply(lambda)` pada teks 100.000 baris?
# Mari kita buktikan secara empiris perbedaan waktu eksekusinya!

# %%
# Membaca data ke Polars DataFrame
df_teks = pl.read_parquet(filepath_parquet)

# 1. Metode A: Python apply(lambda) dengan modul re standar
mulai_lambda = time.time()
sample_lambda = df_teks.head(20_000) # Kita uji pada 20.000 baris saja agar tidak terlalu lama

def python_clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

_ = [python_clean(t) for t in sample_lambda["teks_ulasan"]]
durasi_lambda = time.time() - mulai_lambda
print(f"[Python Interpreter Loop] 20.000 baris selesai dalam: {durasi_lambda:.3f} detik")

# %%
# 2. Metode B: Polars String Expressions (Vectorized Rust Engine)
mulai_polars = time.time()

df_sample_polars = sample_lambda.with_columns(
    pl.col("teks_ulasan")
    .str.to_lowercase()
    .str.replace_all(r"[^a-zA-Z\s]", "")
    .str.replace_all(r"\s+", " ")
    .str.strip_chars()
    .alias("teks_clean")
)

durasi_polars = time.time() - mulai_polars
print(f"[Polars Vectorized Rust] 20.000 baris selesai dalam: {durasi_polars:.3f} detik")
print(f"Percepatan: Polars {durasi_lambda / durasi_polars:.1f}x lebih cepat dibanding Python loop!")

# %% [markdown]
# > ⚡ **Insight Kritis Big Data:**
# > Pembersihan string menggunakan Polars Expressions berjalan secara *multi-threaded* langsung di memori Apache Arrow tanpa serialisasi ke objek Python String. Ini memungkinkan pemrosesan jutaan baris ulasan dalam hitungan detik di laptop!

# %% [markdown]
# ---
# ## Bagian 5: Normalisasi Bahasa Gaul (Slang) & Stopwords Bahasa Indonesia
#
# Kita menerapkan normalisasi kamus gaul marketplace dan menyaring kata-kata tugas yang tidak membawa muatan sentimen dengan tetap **mempertahankan kata negasi**.

# %%
# Kamus penggantian kata gaul / singkatan umum Indonesia
kamus_slang = {
    r"\bbgt\b|\bbangettt\b|\bpol\b": "banget",
    r"\bgk\b|\bgak\b|\bga\b|\btdk\b|\bgada\b": "tidak",
    r"\bcpt\b|\bcepet\b|\bkilat\b|\bgercep\b": "cepat",
    r"\bbrg\b|\bbarange\b": "barang",
    r"\brecomended\b|\brekomen\b|\brecommend\b": "rekomendasi",
    r"\bnyesel\b|\bkecewa\b": "kecewa",
    r"\bsdh\b|\budah\b|\bsudah\b": "sudah",
    r"\bmakasih\b|\bmksih\b|\bthx\b|\bthanks\b": "terimakasih",
    r"\bdgn\b": "dengan",
    r"\byg\b": "yang",
    r"\bori\b|\boriginal\b": "asli",
    r"\bjgn\b": "jangan"
}

# Pipeline pembersihan lengkap pada seluruh 100.000 baris
print("Menjalankan pipeline pembersihan lengkap pada 100.000 ulasan...")
mulai_bersih = time.time()

expr_pembersih = (
    pl.col("teks_ulasan")
    .str.to_lowercase()
    .str.replace_all(r"[^a-zA-Z\s]", " ")
)

# Terapkan penggantian slang regex
for pola, ganti in kamus_slang.items():
    expr_pembersih = expr_pembersih.str.replace_all(pola, ganti)

expr_pembersih = expr_pembersih.str.replace_all(r"\s+", " ").str.strip_chars()

df_clean_all = df_teks.with_columns(expr_pembersih.alias("teks_bersih"))
durasi_total_clean = time.time() - mulai_bersih

print(f"Pembersihan 100.000 ulasan selesai dalam: {durasi_total_clean:.2f} detik!")
print("\nContoh Perbandingan Sebelum dan Sesudah Pembersihan:")
display(df_clean_all.select(["teks_ulasan", "teks_bersih"]).head(5))

# %% [markdown]
# ---
# ## Bagian 6: Ekstraksi Fitur Teks — TF-IDF Vectorization
#
# Kita mengonversi teks yang telah dibersihkan menjadi representasi matriks angka menggunakan **TfidfVectorizer** dari Scikit-learn.
#
# **Parameter Pengendali Memori:**
# - `max_features=5000`: Membatasi jumlah kosa kata terpenting maksimal 5.000 fitur (mencegah OOM).
# - `min_df=5`: Mengabaikan kata yang muncul di kurang dari 5 dokumen ulasan.
# - `ngram_range=(1, 2)`: Menangkap unigram (*"bagus"*) dan bigram (*"tidak bagus"*, *"sangat puas"*).

# %%
# Daftar stopwords bahasa Indonesia (mempertahankan kata negasi: 'tidak', 'jangan', 'bukan', 'kurang')
stopwords_id = [
    "yang", "di", "ke", "dari", "pada", "dalam", "untuk", "dengan", "dan", "atau",
    "ini", "itu", "juga", "sudah", "saya", "kamu", "dia", "mereka", "kita", "kami",
    "ada", "adalah", "lah", "pun", "ya", "nih", "sih", "nya", "gan", "kak", "sis"
]

# Kita filter kelas biner (Positif vs Negatif) untuk klasifikasi fokus
df_biner = df_clean_all.filter(pl.col("sentimen_label").is_in(["positif", "negatif"]))

X_teks_raw = df_biner["teks_bersih"].to_list()
y_label_raw = df_biner["sentimen_label"].to_list()

print(f"Jumlah sampel ulasan biner siap latih: {len(X_teks_raw):,} ulasan")

# Pembagian Data Latih (80%) dan Data Uji (20%) dengan Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X_teks_raw,
    y_label_raw,
    test_size=0.20,
    random_state=2026,
    stratify=y_label_raw
)

print(f"Jumlah data latih : {len(X_train):,} ulasan")
print(f"Jumlah data uji   : {len(X_test):,} ulasan")

# %% [markdown]
# ### 6.1 Memeriksa Kata Kunci Berbobot TF-IDF Tertinggi
# Kita membedah kata-kata yang paling membedakan ulasan positif dan negatif.

# %%
tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    stop_words=stopwords_id,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
print(f"Dimensi Matriks Fitur TF-IDF: {X_train_tfidf.shape}")
print(f"Tipe Struktur Data: {type(X_train_tfidf)} (Compressed Sparse Row)")
print(f"Estimasi Memori Matriks: {X_train_tfidf.data.nbytes / (1024 * 1024):.2f} MB")

# %% [markdown]
# ---
# ## Bagian 7: Pelatihan & Evaluasi Model Klasifikasi Sentimen
#
# Kita melatih model **Logistic Regression** pada matriks fitur TF-IDF untuk memprediksi probabilitas sentimen positif vs negatif.

# %%
# Membangun Pipeline terpadu
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        min_df=5,
        stop_words=stopwords_id,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        C=1.5,
        class_weight="balanced",
        random_state=2026
    ))
])

mulai_latih = time.time()
model_pipeline.fit(X_train, y_train)
durasi_latih = time.time() - mulai_latih

print(f"Model berhasil dilatih dalam: {durasi_latih:.2f} detik!")

# %% [markdown]
# ### 7.1 Evaluasi Kinerja Model pada Data Uji (Classification Report)

# %%
y_pred = model_pipeline.predict(X_test)

print("=== LAPORAN EVALUASI MODEL SENTIMEN (CLASSIFICATION REPORT) ===")
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ### 7.2 Visualisasi Matriks Kebingungan (Confusion Matrix) dengan Plotly

# %%
cm = confusion_matrix(y_test, y_pred, labels=["positif", "negatif"])
labels = ["Positif", "Negatif"]

fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=[f"Prediksi {l}" for l in labels],
    y=[f"Aktual {l}" for l in labels],
    colorscale="Blues",
    text=cm,
    texttemplate="%{text:,}",
    textfont={"size": 16}
))

fig_cm.update_layout(
    title="Confusion Matrix Klasifikasi Sentimen Ulasan E-Commerce",
    xaxis_title="Label Prediksi Model",
    yaxis_title="Label Aktual Ground Truth",
    height=450,
    template="plotly_white"
)
fig_cm.show()

# %% [markdown]
# ### 7.3 Menguji Model pada Kalimat Ulasan Baru (Inference Test)

# %%
kalimat_uji = [
    "Barang sudah mendarat dengan selamat, kualitas juara dan sangat memuaskan!",
    "Kecewa banget, baru dipake 2 jam langsung rusak dan sellernya tidak mau bantu komplain.",
    "Bagus sekali pelayanannya, pengiriman cepat dan harganya terjangkau.",
    "Jangan beli disini, barang palsu tidak sesuai foto penipu!",
    "Bahan lumayan tebal tapi respon seller agak lama, overal oke lah."
]

prediksi_uji = model_pipeline.predict(kalimat_uji)
probabilitas = model_pipeline.predict_proba(kalimat_uji)

print("Hasil Pengujian Sentimen Kalimat Baru:")
for teks, pred, prob in zip(kalimat_uji, prediksi_uji, probabilitas):
    skor_positif = prob[list(model_pipeline.classes_).index("positif")]
    ikon = "✅" if pred == "positif" else "❌"
    print(f"\n{ikon} Teks: \"{teks}\"")
    print(f"   Prediksi : {pred.upper()} (Probabilitas Positif: {skor_positif:.2%})")

# %% [markdown]
# ---
# ## Bagian 8: (Opsional) Hugging Face Transformers & Model IndoNLU
#
# Untuk pemrosesan yang memerlukan pemahaman konteks semantik mendalam, kita dapat memanfaatkan model transformer bahasa Indonesia dari ekosistem Hugging Face.
#
# Di bawah ini adalah contoh kode untuk menguji inferensi model *IndoBERT* menggunakan fungsi `pipeline`:

# %%
try:
    from transformers import pipeline
    print("Transformers terinstal! Menyiapkan pipeline analisis sentimen...")
    
    # Model ulasan bahasa Indonesia dari Hugging Face Hub
    # Catatan: Memerlukan koneksi internet saat pertama kali mengunduh bobot model
    # nlp_transformer = pipeline("sentiment-analysis", model="wirasana/indonesian-sentiment")
    # print(nlp_transformer("Pengiriman cepat dan packing sangat aman, terima kasih seller!"))
    print("Contoh sintaks pipeline IndoBERT siap digunakan jika terhubung ke internet.")
except ImportError:
    print("Catatan: Library 'transformers' dapat diinstal menggunakan 'pip install transformers torch' untuk eksplorasi transformer.")

# %% [markdown]
# ---
# ## Bagian 9: Latihan Mandiri Mahasiswa (Hands-on Challenge)
#
# Kerjakan tiga tantangan analitik di bawah ini untuk menguji pemahaman Anda:

# %% [markdown]
# ### Tantangan 1: Ekstraksi 10 Kata Teratas Pembawa Sentimen Negatif
# Ekstrak koefisien bobot negatif terbesar dari model `LogisticRegression` di dalam pipeline. Tampilkan 10 kata unigram/bigram yang paling kuat mengindikasikan ulasan negatif pelanggan!

# %%
# TULIS KODE TANTANGAN 1 DI SINI
# Petunjuk: Akses model_pipeline.named_steps["classifier"].coef_[0] dan get_feature_names_out()

# %% [markdown]
# ### Tantangan 2: Komparasi Model Multinomial Naive Bayes
# Latih model alternatif `MultinomialNB()` menggunakan matriks TF-IDF yang sama. Bandingkan nilai F1-Score pada kelas negatif antara Naive Bayes dan Logistic Regression!

# %%
# TULIS KODE TANTANGAN 2 DI SINI

# %% [markdown]
# ### Tantangan 3: Agregasi Rasio Sentimen per Kategori Produk via DuckDB
# Menggunakan model yang telah dilatih, prediksikan sentimen pada seluruh data ulasan di `df_clean_all`. Kemudian, gunakan DuckDB untuk menghitung persentase ulasan positif vs negatif untuk setiap kategori produk!

# %%
# TULIS KODE TANTANGAN 3 DI SINI

# %% [markdown]
# ---
# ## Bagian 10: Rangkuman & Checklist Tugas 1 Milestone 3 & Tugas 2
#
# Selamat! Anda telah berhasil menguasai alur kerja pemrosesan teks (NLP) berskala besar dengan Polars dan Scikit-learn.
#
# ### Checklist Milestone Hari Ini:
# - [ ] Memahami keuntungan efisiensi **Polars String Expressions** dibanding Python loop biasa.
# - [ ] Mampu membersihkan teks ulasan bahasa Indonesia dengan kamus *slang* dan menjaga kata negasi.
# - [ ] Berhasil membangun pipeline *TF-IDF* dan melatih model klasifikasi sentimen hemat memori.
# - [ ] Menyelesaikan pelaporan **Tugas 1 Milestone 3 (Time Series & Advanced EDA)** di GitHub Classroom.
# - [ ] Memahami pembagian peran dan skema penilaian **Tugas 2 (Data Product Team Project)**.
#
# Pada **Pertemuan 9**, kita akan mempelajari **Machine Learning pada Big Data** (*Scalable Scikit-learn Pipelines, Memory-efficient Training, Random Forest & Gradient Boosting*)!

