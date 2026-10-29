# ---
# Pertemuan 7: Komputasi Paralel dan Distributed — Dask
# Kuliah Analisis Big Data - Semester Ganjil 2026
# Dosen Pengampu: Yuda Munarko
# ---
#
# Notebook ini adalah panduan hands-on lab untuk Pertemuan 7.
# Silakan jalankan cell di bawah ini secara berurutan.

# %% [markdown]
# # Pertemuan 7: Komputasi Paralel dan Distributed — Dask
#
# ## Capaian Pembelajaran Pertemuan (Sub-CPMK 3 - Kognitif K4):
# 1. Memahami arsitektur komputasi terdistribusi Dask (*Collections*, *Task Graph / DAG*, *Schedulers*, dan *Workers*).
# 2. Menginisialisasi `LocalCluster` dan `Client` dari `dask.distributed` serta menghubungkan dengan **Dask Web Dashboard (Port 8787)**.
# 3. Menerapkan `dask.delayed` untuk memparalelkan fungsi Python kustom dan menganalisis visualisasi *Task Graph*.
# 4. Membangun dan membaca dataset *multi-file Parquet* skala besar secara *out-of-core* menggunakan `dask.dataframe`.
# 5. Mengevaluasi trade-off eksekusi malas (*lazy evaluation*) antara `.compute()` dan `.persist()` pada memori pekerja.
# 6. Mengidentifikasi dampak operasi *shuffling* (seperti `set_index` dan `groupby`) serta strategi *repartitioning* data (~100MB chunk).
# 7. Membandingkan performa *single-node multithreading* (Polars) vs *out-of-core distributed* (Dask).
# 8. Mempersiapkan akuisisi data skala besar untuk checkpoint **Tugas 2 Phase 1**.

# %% [markdown]
# ---
# ## Bagian 1: Inisialisasi Lingkungan Dask Distributed & Web Dashboard
#
# Pada tahap ini, kita menyalakan cluster lokal Dask di dalam container Docker.
# Dask Distributed menyediakan manajemen memori yang aman, *work stealing*, serta **Web Dashboard interaktif pada port 8787**.

# %%
import time
import os
import glob
import numpy as np
import pandas as pd
import polars as pl
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import plotly.express as px
import plotly.graph_objects as go

print(f"Versi Dask: {dask.__version__}")
print(f"Versi Polars: {pl.__version__}")

# Menginisialisasi LocalCluster Dask
# Mengatur batas memori dan jumlah worker secara eksplisit
cluster = LocalCluster(
    n_workers=4,
    threads_per_worker=2,
    memory_limit="2GB",
    dashboard_address=":8787"
)

client = Client(cluster)
print("\nCluster Dask Berhasil Dinyalakan!")
print(f"Dashboard URL: {client.dashboard_link}")
print(client)

# %% [markdown]
# > 💡 **Tips Observabilitas:** Buka browser baru dan akses **[http://localhost:8787](http://localhost:8787)** untuk melihat status cluster, grafik komputasi real-time, dan konsumsi memori saat kode di bawah dijalankan!

# %% [markdown]
# ---
# ## Bagian 2: Paralelisasi Fungsi Kustom dengan `dask.delayed`
#
# Sering kali analitik data melibatkan fungsi Python kustom (misal: scraping data, parsing format file non-standar, atau komputasi matematis spesifik) yang tidak tersedia dalam API DataFrame.
#
# Kita membandingkan eksekusi sekuensial (serial) biasa dengan eksekusi paralel menggunakan `@dask.delayed`.

# %%
# Simulasi fungsi yang memakan waktu I/O atau komputasi (misal: ekstraksi data 0.5 detik)
def proses_sensor(stasiun_id, durasi_detik=0.5):
    time.sleep(durasi_detik)
    nilai_rata = np.random.uniform(25.0, 34.0)
    return {
        "stasiun": stasiun_id,
        "suhu_rata": round(nilai_rata, 2)
    }

def agregasi_seluruh_stasiun(daftar_hasil):
    total = sum(item["suhu_rata"] for item in daftar_hasil)
    rata_rata = total / len(daftar_hasil)
    return round(rata_rata, 2)

daftar_stasiun = [f"STASIUN_{i:02d}" for i in range(1, 9)]
print(f"Daftar stasiun yang akan diproses: {daftar_stasiun}")

# %% [markdown]
# ### 2.1 Eksekusi Sekuensial (Serial) Standar Python
# Seluruh stasiun diproses satu per satu di dalam satu core CPU. Waktu total = 8 x 0.5 detik = ~4 detik.

# %%
mulai_serial = time.time()

hasil_serial = []
for stas in daftar_stasiun:
    hasil_serial.append(proses_sensor(stas, durasi_detik=0.5))

ringkasan_serial = agregasi_seluruh_stasiun(hasil_serial)
durasi_serial = time.time() - mulai_serial

print(f"[Serial] Hasil Agregasi Suhu: {ringkasan_serial}°C")
print(f"[Serial] Waktu Eksekusi: {durasi_serial:.2f} detik")

# %% [markdown]
# ### 2.2 Eksekusi Paralel Menggunakan `dask.delayed`
# Kita membungkus pemanggilan fungsi dengan `dask.delayed`. Operasi ini bersifat *lazy* dan dieksekusi serentak pada multi-worker kluster.

# %%
mulai_paralel = time.time()

# Membungkus fungsi dengan delayed
sensor_delayed = dask.delayed(proses_sensor)
agregasi_delayed = dask.delayed(agregasi_seluruh_stasiun)

# Membangun DAG (Task Graph) - Belum ada eksekusi nyata di sini!
tugas_stasiun = [sensor_delayed(stas, durasi_detik=0.5) for stas in daftar_stasiun]
grafik_agregasi = agregasi_delayed(tugas_stasiun)

print("Representasi Objek Delayed (DAG Belum Dihitung):")
print(grafik_agregasi)

# Memicu eksekusi nyata pada cluster Dask
hasil_paralel = grafik_agregasi.compute()
durasi_paralel = time.time() - mulai_paralel

print(f"\n[Dask Paralel] Hasil Agregasi Suhu: {hasil_paralel}°C")
print(f"[Dask Paralel] Waktu Eksekusi: {durasi_paralel:.2f} detik")
print(f"Percepatan (Speedup): {durasi_serial / durasi_paralel:.1f}x lebih cepat!")

# %% [markdown]
# ### 2.3 Visualisasi Task Graph (DAG)
# Dask dapat merender diagram ketergantungan tugas ke format gambar visual:

# %%
# Render grafik tugas (membutuhkan graphviz)
try:
    grafik_agregasi.visualize(filename="/tmp/dask_delayed_dag.png")
    print("Grafik tugas berhasil divisualisasikan!")
except Exception as e:
    print(f"Catatan visualisasi grafik: {e}")

# %% [markdown]
# ---
# ## Bagian 3: Pembangkitan Multi-File Parquet Sintetis (~1.2 Juta Baris)
#
# Di dunia industri nyata, data berskala besar tidak disimpan dalam 1 file CSV raksasa, melainkan terpecah menjadi puluhan hingga ribuan file berformat **Apache Parquet**.
#
# Kita membuat 6 file partisi transaksi e-commerce Indonesia (masing-masing 200.000 baris = total 1.200.000 baris) di folder `data/processed/dask_ecommerce_demo/`.

# %%
output_dir = "data/processed/dask_ecommerce_demo"
os.makedirs(output_dir, exist_ok=True)

kategori_list = ["Elektronik", "Fashion", "Kebutuhan Rumah", "Makanan & Minuman", "Otomotif", "Kesehatan"]
metode_bayar = ["GoPay", "OVO", "BCA Virtual Account", "ShopeePay", "Kartu Kredit", "COD"]
provinsi_list = ["DKI Jakarta", "Jawa Barat", "Jawa Timur", "Jawa Tengah", "Sumatera Utara", "Sulawesi Selatan"]

np.random.seed(2026)
total_file = 6
baris_per_file = 200_000

print(f"Membangkitkan {total_file} file Parquet ({total_file * baris_per_file:,} total baris)...")

for part_idx in range(total_file):
    filepath = os.path.join(output_dir, f"transaksi_part_{part_idx:02d}.parquet")
    
    # Hanya generate jika file belum ada agar hemat waktu
    if not os.path.exists(filepath):
        order_ids = [f"ORD-{part_idx:02d}-{i:06d}" for i in range(baris_per_file)]
        customer_ids = np.random.randint(100_000, 999_999, size=baris_per_file)
        kategori_sample = np.random.choice(kategori_list, size=baris_per_file)
        metode_sample = np.random.choice(metode_bayar, size=baris_per_file)
        provinsi_sample = np.random.choice(provinsi_list, size=baris_per_file)
        
        harga_dasar = np.round(np.random.exponential(scale=180_000, size=baris_per_file) + 15_000, -2)
        diskon = np.round(np.where(np.random.uniform(0, 1, baris_per_file) > 0.4, harga_dasar * np.random.uniform(0.05, 0.35, baris_per_file), 0), -2)
        ongkir = np.random.choice([0, 10_000, 15_000, 22_000, 35_000], size=baris_per_file)
        total_bayar = harga_dasar - diskon + ongkir
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.03, 0.05, 0.12, 0.35, 0.45], size=baris_per_file)
        
        df_part = pl.DataFrame(
            {
                "order_id": order_ids,
                "customer_id": customer_ids,
                "provinsi": provinsi_sample,
                "kategori": kategori_sample,
                "metode_pembayaran": metode_sample,
                "harga_barang": harga_dasar,
                "diskon": diskon,
                "ongkir": ongkir,
                "total_bayar": total_bayar,
                "rating": rating
            }
        )
        df_part.write_parquet(filepath, compression="zstd")
        print(f"Berhasil menulis: {filepath} ({baris_per_file:,} baris)")
    else:
        print(f"File sudah tersedia: {filepath}")

daftar_file = sorted(glob.glob(f"{output_dir}/*.parquet"))
print(f"\nTotal file Parquet tersedia: {len(daftar_file)} file.")

# %% [markdown]
# ---
# ## Bagian 4: Pembacaan Out-Of-Core & Inspeksi Dask DataFrame
#
# Kita membaca seluruh file Parquet menggunakan wildcard pattern (`*.parquet`).
# Perhatikan bahwa Dask membaca skema (*metadata*) secara instan tanpa memuat 1.2 juta baris ke memori!

# %%
# Membaca seluruh file Parquet sekaligus secara lazy
ddf = dd.read_parquet(f"{output_dir}/*.parquet")

print("Tipe Objek:")
print(type(ddf))

print("\nStruktur Dask DataFrame:")
print(ddf)

print(f"\nJumlah Partisi (npartitions): {ddf.npartitions}")
print(f"Pembagian Indeks (divisions): {ddf.divisions}")

# %% [markdown]
# > 🔍 **Penjelasan Konsep Baru:**
# > - `npartitions`: Jumlah Pandas DataFrame terpisah yang menyusun Dask DataFrame ini. Di sini terdapat 6 partisi (karena berasal dari 6 file Parquet).
# > - `divisions`: Batas indeks antar partisi. Jika nilainya `(None, None, ...)`, artinya data belum diurutkan berdasarkan indeks global (*unknown divisions*).
# > - Pada tahap ini, memori RAM laptop Anda **belum berkurang** karena eksekusi bersifat *lazy*.

# %% [markdown]
# ### 4.1 Mengintip Data Awal dengan `.head()`
# Berbeda dengan `.compute()` yang memproses seluruh data, `.head(n)` hanya mengeksekusi partisi pertama saja sehingga sangat cepat.

# %%
df_contoh = ddf.head(5)
print("5 Baris Pertama Transaksi:")
display(df_contoh)

# %% [markdown]
# ---
# ## Bagian 5: Pipeline Analitik Skala Besar & Eksekusi `.compute()`
#
# Sekarang kita susun pipeline agregasi kompleks:
# 1. Memfilter transaksi bernilai valid (`total_bayar > 50_000` dan `rating >= 3`).
# 2. Menghitung rasio diskon terhadap harga barang.
# 3. Melakukan agregasi per `kategori` dan `provinsi`.
# 4. Menghitung `count`, `mean`, `sum`, dan `max`.

# %%
# 1. Filter dan Penambahan Kolom Baru (Lazy)
ddf_filter = ddf[(ddf["total_bayar"] > 50_000) & (ddf["rating"] >= 3)]
ddf_filter = ddf_filter.assign(
    rasio_diskon=ddf_filter["diskon"] / ddf_filter["harga_barang"]
)

# 2. Groupby Agregasi (Lazy)
agregasi_kategori = ddf_filter.groupby("kategori").agg(
    total_transaksi=("order_id", "count"),
    omset_kotor=("total_bayar", "sum"),
    rata_rata_transaksi=("total_bayar", "mean"),
    rata_rata_diskon=("rasio_diskon", "mean"),
    rata_rata_rating=("rating", "mean")
)

print("Status Objek Agregasi (Masih Lazy):")
print(agregasi_kategori)

# %% [markdown]
# ### 5.1 Memicu Komputasi Terdistribusi dengan `.compute()`
# Buka **Dask Dashboard (port 8787)** sekarang untuk melihat *Task Stream* saat cell di bawah dieksekusi!

# %%
mulai_compute = time.time()

# Memicu eksekusi nyata
df_hasil_kategori = agregasi_kategori.compute()
durasi_compute = time.time() - mulai_compute

print(f"\nKomputasi Selesai dalam {durasi_compute:.2f} detik!")
print("\nHasil Agregasi per Kategori:")
display(df_hasil_kategori.sort_values(by="omset_kotor", ascending=False))

# %% [markdown]
# ### 5.2 Visualisasi Hasil Agregasi dengan Plotly
# Karena hasil dari `.compute()` adalah Pandas DataFrame standar berukuran kecil, kita dapat memvisualisasikannya dengan aman tanpa khawatir browser lag atau crash:

# %%
df_plot = df_hasil_kategori.reset_index()

fig = px.bar(
    df_plot,
    x="kategori",
    y="omset_kotor",
    color="rata_rata_rating",
    text_auto=".2s",
    title="Total Omset Penjualan & Rata-rata Rating per Kategori (Dask Agregasi)",
    labels={
        "kategori": "Kategori Produk",
        "omset_kotor": "Total Omset (Rp)",
        "rata_rata_rating": "Rating Rata-rata"
    },
    color_continuous_scale="Viridis"
)
fig.update_layout(height=450)
fig.show()

# %% [markdown]
# ---
# ## Bagian 6: Menghindari Komputasi Berulang dengan `.persist()`
#
# Perhatikan skenario ini: Anda memiliki dataset yang telah dibersihkan dan difilter, lalu Anda ingin menjalankan 3 kueri analitik berbeda:
# - Kueri A: Omset per metode pembayaran.
# - Kueri B: Omset per provinsi.
# - Kueri C: Distribusi rating kepuasan.
#
# Jika Anda langsung memanggil `.compute()` pada Kueri A, B, dan C, Dask akan **membaca ulang seluruh file Parquet dari disk sebanyak 3 kali!**
#
# Solusinya adalah `.persist()`.

# %%
# Skenario 1: Tanpa persist (Membaca file 2x dari awal)
print("=== Uji Coba 1: TANPA .persist() ===")
mulai_tanpa = time.time()

q1 = ddf_filter.groupby("metode_pembayaran")["total_bayar"].sum().compute()
q2 = ddf_filter.groupby("provinsi")["total_bayar"].sum().compute()

durasi_tanpa = time.time() - mulai_tanpa
print(f"Waktu 2 Kueri Tanpa Persist: {durasi_tanpa:.2f} detik")

# %%
# Skenario 2: DENGAN .persist()
print("\n=== Uji Coba 2: DENGAN .persist() ===")
mulai_dengan = time.time()

# Menyimpan hasil filter di memori RAM pekerja Dask
ddf_filter_cached = ddf_filter.persist()

# Eksekusi kueri dari memori pekerja langsung
q1_fast = ddf_filter_cached.groupby("metode_pembayaran")["total_bayar"].sum().compute()
q2_fast = ddf_filter_cached.groupby("provinsi")["total_bayar"].sum().compute()

durasi_dengan = time.time() - mulai_dengan
print(f"Waktu 2 Kueri Dengan Persist: {durasi_dengan:.2f} detik")
print(f"Peningkatan Kecepatan: {durasi_tanpa / durasi_dengan:.1f}x lebih cepat!")

# %% [markdown]
# ---
# ## Bagian 7: Operasi Shuffling vs Non-Shuffling & Strategi Repartitioning
#
# Mengapa operasi `set_index()` di Dask memakan waktu jauh lebih lama dibanding `assign()` atau `filter()`?
# Jawabannya adalah **Data Shuffling**.

# %%
print(f"Jumlah partisi saat ini: {ddf.npartitions}")

# Mengubah jumlah partisi menjadi 12 partisi (misal untuk kluster dengan 12 core)
ddf_repartitioned = ddf.repartition(npartitions=12)
print(f"Jumlah partisi setelah repartition: {ddf_repartitioned.npartitions}")

# %% [markdown]
# ### 7.1 Menetapkan Index (Global Shuffling)
# Operasi `set_index` mengharuskan data dipindahkan antar pekerja melalui soket TCP agar partisi 0 memuat indeks A-E, partisi 1 memuat F-K, dst.

# %%
mulai_index = time.time()

# Set index pada kolom kategori
ddf_indexed = ddf.set_index("kategori", shuffle="tasks")
contoh_elektronik = ddf_indexed.loc["Elektronik"].head(5)

durasi_index = time.time() - mulai_index
print(f"Waktu Shuffling & Lookup Index: {durasi_index:.2f} detik")
print("Contoh Data Kategori Elektronik via Index Lookup:")
display(contoh_elektronik)

# %% [markdown]
# ---
# ## Bagian 8: Komparasi Kinerja Nyata: Polars vs Dask DataFrame
#
# Kapan sebaiknya kita menggunakan **Polars** dan kapan harus beralih ke **Dask**?
# Mari kita uji pada dataset 1.2 juta baris yang sama.

# %%
# 1. Benchmark Menggunakan Polars (Single-Node In-Memory Multithreading)
mulai_polars = time.time()

polars_hasil = (
    pl.scan_parquet(f"{output_dir}/*.parquet")
    .filter((pl.col("total_bayar") > 50_000) & (pl.col("rating") >= 3))
    .group_by("kategori")
    .agg(
        pl.len().alias("total_transaksi"),
        pl.col("total_bayar").sum().alias("omset_kotor"),
        pl.col("total_bayar").mean().alias("rata_rata_transaksi")
    )
    .collect()
)
durasi_polars = time.time() - mulai_polars

# 2. Benchmark Menggunakan Dask (Out-of-Core Distributed)
mulai_dask = time.time()

dask_hasil = (
    dd.read_parquet(f"{output_dir}/*.parquet")
    .query("total_bayar > 50_000 and rating >= 3")
    .groupby("kategori")
    .agg({
        "total_bayar": ["count", "sum", "mean"]
    })
    .compute()
)
durasi_dask = time.time() - mulai_dask

print(f"Waktu Polars (In-Memory Engine):   {durasi_polars:.3f} detik")
print(f"Waktu Dask   (Distributed Engine): {durasi_dask:.3f} detik")

# %% [markdown]
# > 🧠 **Analisis Insight Kritis (CPMK 3):**
# > Mengapa Polars lebih cepat pada dataset 1.2 juta baris ini?
# > - Dataset 1.2 juta baris (~80MB) masih sepenuhnya muat di dalam RAM laptop.
# > - Polars ditulis dalam bahasa Rust tanpa hambatan Python GIL, mengoptimalkan SIMD CPU dan pembacaan multithreaded murni tanpa overhead komunikasi soket TCP.
# > - Dask memiliki biaya tetap (*fixed overhead*) untuk serialisasi Python, koordinasi scheduler, dan protokol inter-process.
# > - **Kesimpulan:** Dask bukan pengganti Polars untuk data kecil-menengah. Dask bersinar ketika dataset Anda berukuran **100 GB s.d. 10 TB** yang melebihi kapasitas memori fisik satu mesin!

# %% [markdown]
# ---
# ## Bagian 9: Latihan Mandiri Mahasiswa (Hands-on Challenge)
#
# Kerjakan tiga tantangan di bawah ini untuk memperdalam pemahaman praktis Anda:

# %% [markdown]
# ### Tantangan 1: Analisis Loyalitas Pelanggan Berdasarkan Frekuensi Pembelian
# Dengan menggunakan `ddf`, hitunglah:
# 1. 10 `customer_id` dengan total belanja (`total_bayar`) tertinggi di seluruh kategori.
# 2. Berapa rata-rata rating yang diberikan oleh 10 pelanggan teratas tersebut?
# 3. Gunakan `.compute()` hanya pada hasil akhir.

# %%
# TULIS KODE TANTANGAN 1 DI SINI
# Petunjuk: gunakan groupby("customer_id")["total_bayar"].sum()

# %% [markdown]
# ### Tantangan 2: Deteksi Anomali Transaksi (Outlier) per Kategori
# 1. Gunakan `.persist()` pada DataFrame setelah memfilter data di provinsi "DKI Jakarta".
# 2. Hitung nilai persentil ke-95 atau nilai transaksi maksimum per kategori.
# 3. Tampilkan hasilnya dalam bentuk tabel ringkas.

# %%
# TULIS KODE TANTANGAN 2 DI SINI

# %% [markdown]
# ### Tantangan 3: Pemrosesan Paralel dengan `dask.delayed`
# Buatlah fungsi delayed untuk membaca 6 file parquet secara kustom, menghitung jumlah baris unik per file, dan menjumlahkan totalnya menggunakan `dask.compute()`.

# %%
# TULIS KODE TANTANGAN 3 DI SINI

# %% [markdown]
# ---
# ## Bagian 10: Rangkuman & Checklist Tugas 2 Phase 1
#
# Selamat! Anda telah menguasai konsep dasar komputasi paralel dan terdistribusi menggunakan Dask.
#
# ### Checklist Milestone Hari Ini:
# - [ ] Mengakses Dask Dashboard di `http://localhost:8787` dan mengamati *Task Stream*.
# - [ ] Memahami perbedaan fungsi `.compute()` dan `.persist()`.
# - [ ] Memahami aturan ukuran partisi ideal (~100MB per chunk).
# - [ ] Menyelesaikan proposal topik dan skema arsitektur data **Tugas 2 Phase 1** di file `docs/architecture.md`.
#
# Pada **Pertemuan 8**, kita akan melangkah ke materi **Analisis Teks (NLP) pada Data Besar** menggunakan representasi TF-IDF dan model transformer HuggingFace!

# %%
# Menutup client Dask dan membebaskan sumber daya cluster
client.close()
cluster.close()
print("Cluster Dask berhasil dimatikan dengan rapi.")
