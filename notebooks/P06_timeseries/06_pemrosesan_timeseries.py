# ---
# Pertemuan 6: Pemrosesan Data Time Series Besar
# Kuliah Analisis Big Data - Semester Ganjil 2026
# Dosen Pengampu: Yuda Munarko
# ---
#
# Notebook ini adalah panduan hands-on lab untuk Pertemuan 6.
# Silakan jalankan cell di bawah ini secara berurutan.

# %% [markdown]
# # Pertemuan 6: Pemrosesan Data Time Series Besar
#
# ## Capaian Pembelajaran Pertemuan:
# 1. Memahami karakteristik time series big data, timestamp indexing, dan dekomposisi komponen (Trend, Seasonality, Noise).
# 2. Menguasai teknik resampling (downsampling) menggunakan Polars `group_by_dynamic()`.
# 3. Menghitung statistik jendela bergerak (*sliding window*) dengan Polars `rolling_mean()` dan `rolling_std()`.
# 4. Membangun fitur temporal (*Lag, Differencing, Calendar features*) untuk analisis prediktif.
# 5. Mendeteksi dan menangani interval waktu hilang (*gaps*) dengan grid generation, `forward_fill()`, dan `interpolate()`.
# 6. Menerapkan SQL window functions pada DuckDB untuk *running total* dan *cumulative moving window*.
# 7. Membangun arsitektur *Hive Temporal Partitioning* pada Parquet untuk optimasi *query pruning*.
# 8. Membangun visualisasi interaktif deret waktu dengan Plotly (*range slider* dan kalender musiman).

# %% [markdown]
# ---
# ## Bagian 1: Inisialisasi dan Pembuatan Dataset Sensor Cuaca BMKG (~1 Juta Baris)
#
# Kita mensimulasikan data sensor telemetri otomatis (AWS - *Automatic Weather Station*) BMKG
# dari 5 stasiun utama di Indonesia dengan pencatatan berkala setiap menit.

# %%
import duckdb
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import time

# Menetapkan seed agar hasil simulasi konsisten
np.random.seed(2026)

# Daftar Stasiun Meteorologi BMKG
stasiun_info = \
{
    "STAS_01_Kemayoran_Jakarta": (-6.1556, 106.8436),
    "STAS_02_Juanda_Surabaya": (-7.3798, 112.7875),
    "STAS_03_Husein_Bandung": (-6.9006, 107.5761),
    "STAS_04_Kualanamu_Medan": (3.6422, 98.8853),
    "STAS_05_Hasanuddin_Makassar": (-5.0614, 119.5540)
}
daftar_stasiun = list(stasiun_info.keys())

# Membuat rentang waktu 1 tahun dengan interval 1 menit (total 525.600 menit per stasiun)
# Untuk 5 stasiun, total observasi = 5 x 200.000 sampel = 1.000.000 baris
N = 1_000_000

print(f"Membangkitkan {N:,} baris data simulasi sensor time series...")

timestamps = pl.datetime_range(
    pl.datetime(2026, 1, 1, 0, 0, 0),
    pl.datetime(2026, 12, 31, 23, 59, 0),
    interval="1m",
    eager=True
).sample(N, with_replacement=True, seed=2026)

stasiun_kolom = np.random.choice(daftar_stasiun, N)

# Simulasi suhu realistis Indonesia (rata-rata 27°C, dipengaruhi siklus harian dan sedikit noise acak)
# Jam 0-23
jam_array = timestamps.dt.hour().to_numpy()
siklus_suhu = 27.0 + 4.5 * np.sin((jam_array - 9) * 2 * np.pi / 24)
noise_suhu = np.random.normal(0, 0.8, N)
suhu = np.round(siklus_suhu + noise_suhu, 1)

# Kelembaban (berbanding terbalik dengan suhu: 60% - 98%)
kelembaban = np.clip(np.round(100 - (suhu - 20) * 3.2 + np.random.normal(0, 3, N), 1), 40.0, 100.0)

# Curah hujan mm (sebagian besar 0, terkadang hujan sedang/deras 5 - 80 mm)
peluang_hujan = np.random.uniform(0, 1, N)
curah_hujan = np.where(peluang_hujan > 0.88, np.round(np.random.exponential(8.0, N), 1), 0.0)

# Kecepatan angin km/jam (0 - 45 km/jam)
kecepatan_angin = np.round(np.clip(np.random.gamma(shape=3.0, scale=3.5, size=N), 0.5, 55.0), 1)

# Membungkus ke dalam DataFrame Polars
df_raw = pl.DataFrame(
    {
        "timestamp": timestamps,
        "station_id": stasiun_kolom,
        "suhu": suhu,
        "kelembaban": kelembaban,
        "curah_hujan": curah_hujan,
        "kecepatan_angin": kecepatan_angin
    }
).sort(["station_id", "timestamp"])

print(f"Dataset Berhasil Dibuat: {df_raw.shape[0]:,} baris x {df_raw.shape[1]} kolom")
print(df_raw.head(5))

# %% [markdown]
# ---
# ## Bagian 2: Resampling & Downsampling Diskrit dengan Polars `group_by_dynamic()`
#
# Pada data 1 menit, terdapat lebih dari 1.000.000 titik. Jika divisualisasikan langsung ke browser,
# halaman akan mengalami lag (*freeze*).
#
# Kita melakukan agregasi temporal diskrit:
# - Agregasi **per 1 jam**: Menghitung rata-rata suhu, total curah hujan per jam, dan angin puncak.
# - Parameter `every="1h"` mendefinisikan lebar interval bucket.

# %%
t_mulai = time.time()

df_hourly = (
    df_raw
    .sort("timestamp")
    .group_by_dynamic(
        index_column="timestamp",
        every="1h",
        by="station_id",
        closed="left"
    )
    .agg(
        [
            pl.col("suhu").mean().round(2).alias("suhu_rata2"),
            pl.col("suhu").min().alias("suhu_min"),
            pl.col("suhu").max().alias("suhu_maks"),
            pl.col("kelembaban").mean().round(1).alias("kelembaban_rata2"),
            pl.col("curah_hujan").sum().round(1).alias("total_hujan_jam"),
            pl.col("kecepatan_angin").max().alias("kecepatan_angin_maks"),
            pl.len().alias("jumlah_sampel")
        ]
    )
    .sort(["station_id", "timestamp"])
)

durasi_dynamic = time.time() - t_mulai
print(f"Downsampling 1 Juta Baris -> 1 Jam selesai dalam: {durasi_dynamic:.4f} detik!")
print(f"Ukuran Data Terkompresi: {df_hourly.shape[0]:,} baris (Pengurangan ~95% baris)")
print(df_hourly.head(5))

# %% [markdown]
# ---
# ## Bagian 3: Windowing Temporal Halus: Rata-rata Bergerak (`rolling_mean`)
#
# Berbeda dengan `group_by_dynamic()` yang mengecilkan jumlah baris, `rolling()` mempertahankan
# baris dan menghitung statistik lokal pada jendela waktu bergerak (*sliding window*).
#
# Kita akan menghitung:
# - **Moving Average Suhu 24 Jam**: Menghilangkan siklus siang-malam untuk melihat tren musim.
# - **Moving Volatilitas (Std) 24 Jam**: Mengukur stabilitas suhu harian.

# %%
df_stasiun_jkt = df_hourly.filter(pl.col("station_id") == "STAS_01_Kemayoran_Jakarta").sort("timestamp")

df_rolling = df_stasiun_jkt.with_columns(
    [
        # Rata-rata bergerak 24 jam (24 baris data per jam)
        pl.col("suhu_rata2")
        .rolling_mean(window_size=24)
        .round(2)
        .alias("suhu_ma_24h"),

        # Standar deviasi bergerak 24 jam
        pl.col("suhu_rata2")
        .rolling_std(window_size=24)
        .round(2)
        .alias("suhu_std_24h"),

        # Hujan kumulatif bergerak 3 hari (72 jam)
        pl.col("total_hujan_jam")
        .rolling_sum(window_size=72)
        .round(1)
        .alias("hujan_kumulatif_72h")
    ]
)

print("Tabel Fitur Rolling Window (Jakarta):")
print(df_rolling.select(["timestamp", "suhu_rata2", "suhu_ma_24h", "suhu_std_24h", "hujan_kumulatif_72h"]).head(10))

# %% [markdown]
# ---
# ## Bagian 4: Feature Engineering Temporal (Lag, Differencing, & Kalender)
#
# Dalam analisis prediksi deret waktu, kita membutuhkan:
# 1. **Lag 1 Jam ($Y_{t-1}$)**: Suhu 1 jam yang lalu.
# 2. **Lag 24 Jam ($Y_{t-24}$)**: Suhu pada jam yang sama di hari kemarin (kunci menangkap musiman).
# 3. **Differencing 1 Jam ($\Delta Y = Y_t - Y_{t-1}$)**: Laju kenaikan atau penurunan suhu.
# 4. **Fitur Kalender**: Jam, hari dalam seminggu, bulan, dan penanda akhir pekan (*weekend*).

# %%
df_features = df_stasiun_jkt.with_columns(
    [
        # Lag 1 jam
        pl.col("suhu_rata2").shift(1).alias("suhu_lag_1h"),

        # Lag 24 jam (musiman harian)
        pl.col("suhu_rata2").shift(24).alias("suhu_lag_24h"),

        # Perubahan nilai (differencing)
        pl.col("suhu_rata2").diff(1).round(2).alias("suhu_diff_1h"),

        # Ekstraksi fitur waktu
        pl.col("timestamp").dt.hour().alias("jam"),
        pl.col("timestamp").dt.weekday().alias("hari_ke"),  # 1 = Senin, 7 = Minggu
        pl.col("timestamp").dt.month().alias("bulan"),
        (pl.col("timestamp").dt.weekday() >= 6).alias("is_weekend")
    ]
)

print("Fitur Temporal yang Terbentuk:")
print(df_features.select(["timestamp", "suhu_rata2", "suhu_lag_1h", "suhu_lag_24h", "suhu_diff_1h", "jam", "hari_ke", "is_weekend"]).head(8))

# %% [markdown]
# ---
# ## Bagian 5: Penanganan Celah Waktu (*Gaps / Missing Timestamps*)
#
# Dalam dunia nyata, stasiun cuaca dapat mati sesaat karena kendala daya atau jaringan.
# Hal ini menghasilkan **jeda waktu kosong** yang merusak kalkulasi `shift()` dan `rolling()`.
#
# **Standar Penanganan:**
# 1. Buat grid waktu lengkap (`pl.datetime_range`).
# 2. Left join grid dengan data sensor.
# 3. Imputasi: `forward_fill()` untuk data diskrit, `interpolate()` untuk suhu/kontinu.

# %%
# Mensimulasikan data sensor yang bolong (menghapus baris jam 10:00 - 14:00 pada tanggal 5 Januari 2026)
waktu_rusak_mulai = pl.datetime(2026, 1, 5, 10, 0, 0)
waktu_rusak_akhir = pl.datetime(2026, 1, 5, 14, 0, 0)

df_bolong = df_stasiun_jkt.filter(
    (pl.col("timestamp") < waktu_rusak_mulai) | (pl.col("timestamp") > waktu_rusak_akhir)
)

print(f"Data Sebelum Gap: {df_stasiun_jkt.shape[0]} baris")
print(f"Data Mengalami Gap: {df_bolong.shape[0]} baris (Ada 5 baris jam hilang)")

# 1. Membangun Grid Waktu Lengkap
t_min = df_bolong["timestamp"].min()
t_max = df_bolong["timestamp"].max()

grid_lengkap = pl.DataFrame(
    {
        "timestamp": pl.datetime_range(t_min, t_max, interval="1h", eager=True)
    }
)

# 2. Gabungkan Grid dengan Data Bolong
df_restored = (
    grid_lengkap
    .join(df_bolong, on="timestamp", how="left")
    .sort("timestamp")
    .with_columns(
        [
            # Forward fill: mengisi dengan nilai sebelum gap
            pl.col("suhu_rata2").forward_fill().alias("suhu_ffill"),

            # Interpolasi linear: membuat garis mulus melewati titik yang hilang
            pl.col("suhu_rata2").interpolate().round(2).alias("suhu_interpolasi")
        ]
    )
)

# Menampilkan baris di sekitar gap tanggal 5 Januari
gap_view = df_restored.filter(
    (pl.col("timestamp") >= pl.datetime(2026, 1, 5, 8, 0, 0)) &
    (pl.col("timestamp") <= pl.datetime(2026, 1, 5, 16, 0, 0))
)
print("Hasil Imputasi pada Jeda Sensor Mati:")
print(gap_view.select(["timestamp", "suhu_rata2", "suhu_ffill", "suhu_interpolasi"]))

# %% [markdown]
# ---
# ## Bagian 6: Analisis Time Series dengan SQL DuckDB Window Functions
#
# DuckDB memiliki dukungan lengkap untuk klausa `OVER (ORDER BY ... ROWS BETWEEN ...)`
# yang sangat fleksibel untuk menghitung:
# 1. **Running Total Kumulatif Curah Hujan**: Akumulasi hujan sepanjang tahun.
# 2. **Peringkat Suhu Harian (*Rank*)**: Mendeteksi jam-jam terpanas di setiap stasiun.

# %%
con = duckdb.connect()
con.register("sensor_hourly", df_hourly)

query_sql = """
    SELECT
        station_id,
        timestamp,
        suhu_rata2,
        total_hujan_jam,
        -- 1. Running Total Kumulatif Hujan per Stasiun
        SUM(total_hujan_jam) OVER (
            PARTITION BY station_id
            ORDER BY timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS curah_hujan_kumulatif,

        -- 2. Rata-rata Suhu 3 Jam (1 baris sebelum, baris aktif, 1 baris sesudah)
        AVG(suhu_rata2) OVER (
            PARTITION BY station_id
            ORDER BY timestamp
            ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        ) AS suhu_moving_3h
    FROM sensor_hourly
    ORDER BY station_id, timestamp
    LIMIT 10;
"""

hasil_duckdb = con.execute(query_sql).pl()
print("Hasil SQL Window Function DuckDB:")
print(hasil_duckdb)

# %% [markdown]
# ---
# ## Bagian 7: Hive Temporal Partitioning Parquet & Query Pruning
#
# Ketika data mencapai puluhan gigabyte, menyimpan semua data dalam 1 file tunggal membuat kueri lambat.
#
# Kita menerapkan partisi direktori berbasis waktu:
# `data/cuaca_partitioned/tahun=YYYY/bulan=MM/*.parquet`
#
# Kueri dengan filter `WHERE tahun = 2026 AND bulan = 3` hanya akan membaca folder bulan 3,
# dan **mengabaikan 91% data lainnya** tanpa membaca disk (*Partition Pruning*).

# %%
import os
import shutil

folder_output = "/home/jovyan/work/data/cuaca_partitioned"

# Bersihkan jika sebelumnya terbuat sebagai file atau direktori lama
if os.path.exists(folder_output):
    if os.path.isdir(folder_output):
        shutil.rmtree(folder_output)
    else:
        os.remove(folder_output)

os.makedirs("/home/jovyan/work/data", exist_ok=True)

# Menambahkan kolom tahun dan bulan untuk partisi
df_partisi_siap = df_hourly.with_columns(
    [
        pl.col("timestamp").dt.year().alias("tahun"),
        pl.col("timestamp").dt.month().alias("bulan")
    ]
)

print("Mengekspor dataset dengan Hive Temporal Partitioning via DuckDB...")
con.register("df_partisi_siap", df_partisi_siap)
con.execute(f"""
    COPY df_partisi_siap
    TO '{folder_output}'
    (FORMAT PARQUET, PARTITION_BY (tahun, bulan), OVERWRITE_OR_IGNORE 1);
""")
print(f"Data berhasil dipartisi ke folder: {folder_output}")

# Kueri DuckDB langsung ke file partisi dengan wildcard fleksibel
t_pruning = time.time()
kueri_pruning = con.execute(f"""
    SELECT
        station_id,
        AVG(suhu_rata2) AS suhu_maret,
        SUM(total_hujan_jam) AS total_hujan_maret
    FROM '{folder_output}/**/*.parquet'
    WHERE tahun = 2026 AND bulan = 3
    GROUP BY station_id
    ORDER BY total_hujan_maret DESC
""").pl()
durasi_pruning = time.time() - t_pruning

print(f"Kueri Bulan Maret selesai dalam: {durasi_pruning:.4f} detik dengan Partition Pruning!")
print(kueri_pruning)

# %% [markdown]
# ---
# ## Bagian 8: Visualisasi Interaktif Deret Waktu dengan Plotly
#
# Kita membuat 2 visualisasi representatif:
# 1. **Line Chart dengan Range Slider**: Menampilkan suhu aktual vs Moving Average 24 jam dengan interaktivitas zoom rentang tanggal.
# 2. **Siklus Musiman Harian**: Distribusi suhu rata-rata per jam di setiap stasiun.

# %%
# Mengambil sampel data 1 bulan untuk visualisasi garis interaktif
df_plot_bulan1 = (
    df_rolling
    .filter(pl.col("timestamp") <= pl.datetime(2026, 1, 31, 23, 59, 0))
    .to_pandas()
)

fig = go.Figure()

# Garis Suhu Aktual
fig.add_trace(
    go.Scatter(
        x=df_plot_bulan1["timestamp"],
        y=df_plot_bulan1["suhu_rata2"],
        mode="lines",
        name="Suhu Riil Per Jam",
        line=dict(color="#93c5fd", width=1),
        opacity=0.7
    )
)

# Garis Moving Average 24 Jam
fig.add_trace(
    go.Scatter(
        x=df_plot_bulan1["timestamp"],
        y=df_plot_bulan1["suhu_ma_24h"],
        mode="lines",
        name="Moving Average 24 Jam (Tren)",
        line=dict(color="#1d4ed8", width=2.5)
    )
)

fig.update_layout(
    title="Dinamika Suhu BMKG Jakarta — Januari 2026 (Fluktuasi Harian vs Tren)",
    xaxis_title="Waktu",
    yaxis_title="Suhu (°C)",
    hovermode="x unified",
    template="plotly_white",
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1 Hari", step="day", stepmode="backward"),
                    dict(count=7, label="7 Hari", step="day", stepmode="backward"),
                    dict(step="all", label="1 Bulan Lengkap")
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)
fig.show()

# %% [markdown]
# ---
# ## Bagian 9: Evaluasi Mandiri Mahasiswa (Latihan Lab)
#
# Kerjakan latihan mandiri berikut untuk memantapkan pemahaman Anda:
#
# 1. **Latihan 1 (Dynamic Resampling)**:
#    Lakukan *downsampling* pada `df_raw` menjadi agregasi **harian (`every="1d"`)**, lalu hitung selisih suhu maksimum dan minimum (*diurnal temperature range*) per hari di setiap stasiun.
#
# 2. **Latihan 2 (Lag Feature & Differencing)**:
#    Pada dataset harian tersebut, buat fitur `suhu_lag_7d` (suhu 1 minggu sebelumnya) dan `delta_hujan_mingguan` menggunakan Polars expressions.
#
# 3. **Latihan 3 (DuckDB SQL)**:
#    Tuliskan kueri DuckDB untuk menemukan 3 stasiun dengan total curah hujan tertinggi di bulan tertentu menggunakan data partisi Parquet.
