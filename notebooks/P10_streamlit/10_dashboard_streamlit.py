# ---
# Pertemuan 10: Membangun Dashboard Data — Streamlit
# Kuliah Analisis Big Data - Semester Ganjil 2026
# Dosen Pengampu: Yuda Munarko
# ---
#
# Notebook ini adalah panduan hands-on lab untuk Pertemuan 10.
# Silakan jalankan cell di bawah ini secara berurutan.

# %% [markdown]
# # Pertemuan 10: Membangun Dashboard Data — Streamlit
#
# ## Capaian Pembelajaran Pertemuan (Sub-CPMK 4 - Kognitif K3, K6):
# 1. Menjelaskan **model eksekusi reaktif** Streamlit: seluruh skrip dijalankan ulang setiap ada interaksi.
# 2. Merancang **lapisan kueri (backend)** dengan DuckDB sehingga dashboard hanya menerima hasil ringkas, bukan jutaan baris mentah.
# 3. Menulis filter **berparameter** (`?`) yang aman dari SQL injection.
# 4. Menerapkan **caching** yang tepat: `@st.cache_data` untuk hasil kueri, `@st.cache_resource` untuk koneksi/model.
# 5. Membangun dashboard interaktif: sidebar filter, `st.form`, KPI `st.metric`, `st.tabs`, dan grafik Plotly.
# 6. Menyiapkan **deployment ke HuggingFace Spaces** (Docker SDK) sebagai bekal Tugas 2 Phase 2–3.
#
# ## Di mana Streamlit dijalankan?
# Streamlit **bukan** dijalankan di dalam notebook. Pembagian perannya:
#
# | Bagian | Tempat | Fungsi |
# |---|---|---|
# | Notebook ini | JupyterLab | Menyiapkan data, menguji kueri & mengukur performa |
# | `queries.py` | Modul Python | Lapisan kueri DuckDB (dipakai notebook **dan** dashboard) |
# | `app.py` | Terminal → `streamlit run` | Dashboard interaktif di `http://localhost:8501` |
# | HuggingFace Spaces | Cloud | Dashboard *live* untuk publik (Tugas 2) |
#
# Pola ini sama dengan template Tugas 2: `src/queries.py` + `app.py`.

# %% [markdown]
# ---
# ## Bagian 1: Inisialisasi Lingkungan

# %%
import os
import time
from datetime import date

import duckdb
import plotly.express as px
import polars as pl

import queries as q

print(f"Versi DuckDB : {duckdb.__version__}")
print(f"Versi Polars : {pl.__version__}")
try:
    import streamlit as st
    print(f"Versi Streamlit: {st.__version__}")
except ImportError:
    print("Streamlit belum terpasang. Rebuild image Docker: docker compose up --build")

# %% [markdown]
# ---
# ## Bagian 2: Dataset Transaksi E-Commerce 2025 (2 Juta Baris)
#
# Dataset sintetis dibangkitkan **langsung oleh DuckDB ke Parquet** (tanpa melewati Pandas) melalui `q.siapkan_data()`:
# - 10 provinsi, 6 kategori, 5 metode pembayaran.
# - Pola realistis: tren naik sepanjang tahun, lonjakan **tanggal kembar** (11.11, 12.12), akhir pekan lebih ramai, puncak belanja **pukul 19–22**.
# - 8% transaksi **belum diberi rating** (NULL).
# - Data diurutkan berdasarkan waktu sehingga filter tanggal dapat melompati *row group*.

# %%
t0 = time.perf_counter()
path = q.siapkan_data()
print(f"File   : {path}")
print(f"Ukuran : {os.path.getsize(path) / 1024**2:.1f} MB (Parquet zstd)")
print(f"Waktu  : {time.perf_counter() - t0:.1f} detik (0 detik jika file sudah ada)")

con = q.buat_koneksi()
con.sql("DESCRIBE transaksi").show()
con.sql("SELECT * FROM transaksi LIMIT 5").show()

# %% [markdown]
# ### Metadata Parquet: statistik min/max per row group
# Setiap *row group* menyimpan nilai **min/max** tiap kolom. Saat kita memfilter `tanggal >= '2025-12-01'`, DuckDB cukup membaca row group yang rentang waktunya cocok. Inilah alasan data diurutkan berdasarkan waktu.

# %%
con.sql(f"""
    SELECT row_group_id, row_group_num_rows,
           stats_min::TIMESTAMP AS waktu_min, stats_max::TIMESTAMP AS waktu_max
    FROM parquet_metadata('{path.as_posix()}')
    WHERE path_in_schema = 'waktu'
    ORDER BY row_group_id
""").show(max_rows=20)

# %% [markdown]
# ---
# ## Bagian 3: Kenapa Dashboard Tidak Boleh Memuat Seluruh Data?
#
# Pada Streamlit, **setiap klik** filter menjalankan ulang skrip. Bandingkan dua strategi untuk menjawab pertanyaan *"berapa omzet per wilayah di bulan Desember?"*:
#
# - **Strategi A (anti-pattern):** baca seluruh Parquet ke DataFrame, lalu filter di Python.
# - **Strategi B (dianjurkan):** kirim SQL ke DuckDB; yang kembali hanya 10 baris hasil agregasi.

# %%
# Strategi A: muat semua baris, baru difilter
t0 = time.perf_counter()
df_semua = pl.read_parquet(path)
hasil_a = (
    df_semua.filter(pl.col("waktu").dt.month() == 12)
    .group_by("wilayah")
    .agg((pl.col("jumlah") * pl.col("harga_satuan")).sum().alias("omzet"))
)
waktu_a = time.perf_counter() - t0
mem_a = df_semua.estimated_size("mb")

# Strategi B: agregasi di DuckDB, langsung dari Parquet
t0 = time.perf_counter()
where, params = q.bangun_filter(tgl_awal=date(2025, 12, 1))
hasil_b = q.query_per_dimensi(con, where, params, "wilayah")
waktu_b = time.perf_counter() - t0
mem_b = hasil_b.estimated_size("mb")

print(f"A. Muat semua lalu filter : {waktu_a:6.2f} s, data di memori {mem_a:8.1f} MB")
print(f"B. Agregasi di DuckDB     : {waktu_b:6.2f} s, data di memori {mem_b:8.4f} MB")
del df_semua

# %% [markdown]
# Bayangkan Strategi A dijalankan **setiap kali** pengguna menggeser slider, oleh **puluhan pengguna sekaligus**, di server HuggingFace Spaces gratis (2 vCPU, 16 GB RAM). Dashboard akan lambat lalu mati karena *Out-of-Memory*.
#
# > **Prinsip:** dashboard hanya menerima **data seukuran grafik**. Grafik garis 365 titik tidak membutuhkan 2 juta baris.

# %% [markdown]
# ---
# ## Bagian 4: Filter Berparameter (Aman dari SQL Injection)
#
# Nilai filter berasal dari pengguna. **Jangan** menyisipkannya ke SQL dengan f-string.

# %%
# BERBAHAYA: input pengguna disisipkan langsung ke SQL
input_pengguna = "Bali' OR '1'='1"
sql_bahaya = f"SELECT count(*) FROM transaksi WHERE wilayah = '{input_pengguna}'"
print("f-string  :", con.execute(sql_bahaya).fetchone()[0], "baris  <- filter jebol, semua baris terbaca")

# AMAN: nilai dikirim sebagai parameter (?)
print("parameter :", con.execute("SELECT count(*) FROM transaksi WHERE wilayah = ?", [input_pengguna]).fetchone()[0], "baris")

# %% [markdown]
# `q.bangun_filter()` menyusun klausa `WHERE` beserta daftar parameternya dari pilihan widget:

# %%
where, params = q.bangun_filter(
    wilayah=["DKI Jakarta", "Bali"], kategori=["Elektronik"],
    tgl_awal=None, tgl_akhir=None,
)
print(where)
print(params)
q.query_kpi(con, where, params)

# %% [markdown]
# ---
# ## Bagian 5: Model Eksekusi Streamlit & Caching
#
# Streamlit menjalankan `app.py` **dari baris pertama sampai terakhir** setiap kali:
# 1. Pengguna membuka halaman.
# 2. Pengguna mengubah widget apa pun (multiselect, slider, tombol, tab radio, ...).
#
# Akibatnya, fungsi pemuatan data ikut dijalankan ulang **kecuali di-cache**. Cell berikut mensimulasikan 5 interaksi pengguna.

# %%
def skrip_dashboard(wilayah: tuple):
    """Simulasi isi app.py: semua kueri untuk satu kombinasi filter."""
    where, params = q.bangun_filter(wilayah=wilayah)
    return {
        "kpi": q.query_kpi(con, where, params),
        "tren": q.query_tren(con, where, params, "day"),
        "kategori": q.query_per_dimensi(con, where, params, "kategori"),
        "pola": q.query_pola_jam(con, where, params),
    }


interaksi = [("Bali",), ("DKI Jakarta",), ("Bali",), ("Bali",), ("DKI Jakarta",)]

# Tanpa cache
t0 = time.perf_counter()
for w in interaksi:
    skrip_dashboard(w)
print(f"Tanpa cache : {time.perf_counter() - t0:.2f} s untuk {len(interaksi)} interaksi")

# Dengan cache sederhana: kunci = argumen fungsi (inilah prinsip @st.cache_data)
cache = {}
def skrip_dashboard_cached(wilayah: tuple):
    if wilayah not in cache:
        cache[wilayah] = skrip_dashboard(wilayah)   # cache miss: jalankan kueri
    return cache[wilayah]                           # cache hit: langsung kembali

t0 = time.perf_counter()
for w in interaksi:
    skrip_dashboard_cached(w)
print(f"Dengan cache: {time.perf_counter() - t0:.2f} s (hanya {len(cache)} kombinasi unik yang dihitung)")

# %% [markdown]
# ### `@st.cache_data` vs `@st.cache_resource`
#
# | | `@st.cache_data` | `@st.cache_resource` |
# |---|---|---|
# | Untuk | Hasil yang bisa diserialisasi: DataFrame, dict, list | Objek "hidup": koneksi DuckDB, model ML, klien API |
# | Penyimpanan | **Salinan** per pemanggilan (aman diubah) | **Satu objek** dipakai bersama semua pengguna |
# | Kunci cache | Isi argumen fungsi + kode fungsi | Isi argumen fungsi + kode fungsi |
# | Opsi penting | `ttl="1h"`, `max_entries=100` | `ttl`, `max_entries` |
#
# Di `app.py`:
# ```python
# @st.cache_resource              # dibuat sekali per server
# def get_koneksi():
#     return q.buat_koneksi()
#
# @st.cache_data(ttl="1h", max_entries=100)   # per kombinasi filter
# def hitung_dashboard(wilayah: tuple, kategori: tuple, tgl_awal, tgl_akhir, granularitas): ...
# ```
#
# **Catatan thread:** setiap sesi pengguna berjalan di *thread* berbeda, sehingga koneksi DuckDB yang dipakai bersama tidak boleh dipakai langsung. `queries._jalankan()` membuat `con.cursor()` untuk setiap kueri.

# %% [markdown]
# ---
# ## Bagian 6: Pratinjau Grafik Dashboard
#
# Grafik yang sama dengan `app.py`, diuji dulu di notebook. Ukuran data yang dikirim ke Plotly hanya ratusan baris.

# %%
tren = q.query_tren(con, "", [], "day")
print(f"Baris untuk grafik tren harian: {tren.height}")
fig = px.line(tren, x="periode", y="omzet", title="Omzet harian 2025: perhatikan 11.11 dan 12.12")
fig.show()

# %%
pola = q.query_pola_jam(con, "", []).with_columns(
    pl.col("hari").replace_strict({1: "Senin", 2: "Selasa", 3: "Rabu", 4: "Kamis", 5: "Jumat", 6: "Sabtu", 7: "Minggu"},
                                  return_dtype=pl.String).alias("nama_hari")
)
fig = px.density_heatmap(pola, x="jam", y="nama_hari", z="jumlah_transaksi", histfunc="sum", nbinsx=24,
                         category_orders={"nama_hari": ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]},
                         color_continuous_scale="Blues", title="Pola transaksi per hari dan jam")
fig.show()

# %% [markdown]
# ---
# ## Bagian 7: Menjalankan Dashboard
#
# Streamlit sudah terpasang di container Docker kelas. Perintah `streamlit` **tidak ada di terminal laptop**, jadi jalankan dengan salah satu cara berikut.
#
# **Cara 1: Terminal JupyterLab** (*File → New → Terminal*):
#
# ```bash
# cd ~/work/notebooks/P10_streamlit
# streamlit run app.py --server.port 8501 --server.address 0.0.0.0
# ```
#
# **Cara 2: Terminal laptop** (dari folder `bigdata-kuliah`):
#
# ```bash
# docker compose exec -w /home/jovyan/work/notebooks/P10_streamlit jupyter \
#     streamlit run app.py --server.port 8501 --server.address 0.0.0.0
# ```
#
# Buka **http://localhost:8501** di browser laptop (port 8501 sudah dipetakan di `docker-compose.yml`). Hentikan dengan `Ctrl+C` di terminal.
#
# ### Anatomi `app.py`
# 1. `st.set_page_config(layout="wide")` — wajib baris Streamlit pertama.
# 2. `@st.cache_resource get_koneksi()` dan `@st.cache_data hitung_dashboard(...)`.
# 3. **Sidebar** berisi `st.form`: perubahan filter baru diproses saat tombol *Terapkan filter* ditekan (menghemat rerun).
# 4. `st.spinner(...)` saat kueri berjalan.
# 5. **KPI** dengan `st.metric` (+ delta vs periode sebelumnya) di baris atas.
# 6. `st.tabs`: Tren, Wilayah & Kategori, Pola Waktu, Data.
# 7. `st.dataframe` dengan `column_config` dan `st.download_button`.
#
# ### Eksperimen cache
# 1. Pilih filter, tekan *Terapkan*, catat **waktu ambil data** di bagian bawah.
# 2. Ganti ke filter lain lalu kembali ke filter pertama: waktunya mendekati 0 ms (*cache hit*).
# 3. Nyalakan **Mode demo: matikan cache** dan ulangi. Setiap interaksi kini menjalankan 8 kueri.

# %% [markdown]
# ---
# ## Bagian 8: Deployment ke HuggingFace Spaces (Bekal Tugas 2)
#
# 1. Buat akun di [huggingface.co](https://huggingface.co) → **New Space** → SDK **Docker** → template *Blank*.
# 2. Isi repositori Space dengan `app.py`, modul kueri, `requirements.txt`, dan `Dockerfile` (contoh lengkap ada di template Tugas 2).
# 3. Blok YAML di baris paling atas `README.md` Space:
#    ```yaml
#    ---
#    title: Dashboard Penjualan
#    sdk: docker
#    app_port: 8501
#    ---
#    ```
# 4. `git push` ke repo Space; build berjalan otomatis (lihat tab **Logs**).
#
# **Strategi data di cloud** (file > 100 MB tidak boleh di-commit ke GitHub):
# - Simpan **Parquet hasil agregasi/pembersihan**, bukan data mentah.
# - Unggah Parquet ke **HuggingFace Datasets**, lalu baca langsung dengan DuckDB: `read_parquet('hf://datasets/<user>/<repo>/file.parquet')`.
# - Atau bangkitkan/unduh data saat aplikasi pertama kali berjalan (seperti `q.siapkan_data()`), dibungkus `@st.cache_resource`.
#
# Space gratis: 2 vCPU, 16 GB RAM, dan akan *sleep* bila lama tidak diakses.

# %% [markdown]
# ---
# ## Bagian 9: Latihan Mandiri (Hands-on Challenge)
#
# Kerjakan di `app.py` (dan `queries.py` bila perlu), lalu amati hasilnya di browser.

# %% [markdown]
# ### Tantangan 1: Filter Metode Pembayaran
# Tambahkan `st.multiselect("Metode pembayaran", ...)` di dalam form, perluas `q.bangun_filter()` dengan parameter `metode_bayar`, dan teruskan ke `hitung_dashboard`. Pastikan tetap memakai parameter `?`.

# %%
# TODO: uji fungsi bangun_filter versi baru di sini sebelum dipakai di app.py


# %% [markdown]
# ### Tantangan 2: KPI Persentase COD
# Tambahkan kueri yang menghitung **persentase transaksi COD** dan tampilkan sebagai `st.metric` kelima. Apakah wilayah luar Jawa lebih banyak memakai COD?

# %%
# TODO: tulis dan uji kuerinya di sini


# %% [markdown]
# ### Tantangan 3: Ukur Dampak Cache
# Dengan Mode demo, catat waktu ambil data untuk 3 filter berbeda, masing-masing **dengan** dan **tanpa** cache. Sajikan dalam tabel kecil dan jelaskan mengapa `max_entries` penting jika ada ribuan kombinasi filter.

# %%
# TODO: catat hasil pengukuran Anda di sini


# %% [markdown]
# ---
# ## Rangkuman
# 1. Streamlit **menjalankan ulang seluruh skrip** pada setiap interaksi; kerja berat wajib di-cache.
# 2. Pisahkan **lapisan kueri** (`queries.py`, DuckDB) dari **lapisan tampilan** (`app.py`, Streamlit).
# 3. Dashboard hanya menerima **data seukuran grafik**: agregasi, `LIMIT`, dan filter dilakukan di DuckDB.
# 4. Gunakan **parameter `?`** untuk nilai dari widget, bukan f-string.
# 5. `@st.cache_data` untuk hasil kueri (`ttl`, `max_entries`); `@st.cache_resource` untuk koneksi dan model.
# 6. `st.form`, `st.spinner`, `st.metric`, dan `st.tabs` membuat dashboard hemat rerun dan mudah dibaca.
#
# **Tugas 2:** check-in Phase 1 hari ini, lalu **Phase 2 (Pertemuan 11):** pindahkan pola `queries.py` + caching ini ke `src/queries.py` tim dengan dataset kalian.
