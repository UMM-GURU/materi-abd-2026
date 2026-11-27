"""Dashboard Penjualan E-Commerce 2025 — Pertemuan 10 (Streamlit)

Streamlit sudah terpasang di container Docker kelas (tidak di terminal laptop).

Cara 1, dari terminal JupyterLab:
    cd ~/work/notebooks/P10_streamlit
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0

Cara 2, dari terminal laptop (di folder bigdata-kuliah):
    docker compose exec -w /home/jovyan/work/notebooks/P10_streamlit jupyter \
        streamlit run app.py --server.port 8501 --server.address 0.0.0.0

Lalu buka http://localhost:8501 di browser laptop.

Ingat model eksekusi Streamlit: SELURUH skrip ini dijalankan ulang dari atas
setiap kali pengguna berinteraksi. Karena itu kerja berat wajib di-cache.
"""

import time
from datetime import timedelta

import plotly.express as px
import polars as pl
import streamlit as st

import queries as q

st.set_page_config(page_title="Dashboard Penjualan E-Commerce 2025", page_icon="🛒", layout="wide")

NAMA_HARI = {1: "Senin", 2: "Selasa", 3: "Rabu", 4: "Kamis", 5: "Jumat", 6: "Sabtu", 7: "Minggu"}
GRANULARITAS = {"Harian": "day", "Mingguan": "week", "Bulanan": "month"}


def rupiah(x: float | None) -> str:
    if x is None:
        return "-"
    for batas, satuan in [(1e12, "T"), (1e9, "M"), (1e6, "jt"), (1e3, "rb")]:
        if abs(x) >= batas:
            return f"Rp {x / batas:,.2f} {satuan}".replace(".", ",")
    return f"Rp {x:,.0f}"


# ---------------------------------------------------------------------------
# 1. Caching
# ---------------------------------------------------------------------------
@st.cache_resource
def get_koneksi():
    """RESOURCE: dibuat sekali per proses server lalu dipakai bersama oleh semua pengguna.
    (Koneksi database, model ML. Objeknya tidak disalin.)"""
    return q.buat_koneksi()


def hitung_dashboard(wilayah: tuple, kategori: tuple, tgl_awal, tgl_akhir, granularitas: str) -> dict:
    """Semua kueri untuk satu kombinasi filter. Hasilnya kecil (ringkasan), bukan data mentah."""
    con = get_koneksi()
    where, params = q.bangun_filter(wilayah, kategori, tgl_awal, tgl_akhir)

    # Periode sebelumnya dengan panjang yang sama, untuk delta pada st.metric
    panjang = tgl_akhir - tgl_awal
    where_lalu, params_lalu = q.bangun_filter(wilayah, kategori, tgl_awal - panjang - timedelta(days=1),
                                              tgl_awal - timedelta(days=1))
    return {
        "kpi": q.query_kpi(con, where, params),
        "kpi_lalu": q.query_kpi(con, where_lalu, params_lalu),
        "tren": q.query_tren(con, where, params, granularitas),
        "wilayah": q.query_per_dimensi(con, where, params, "wilayah"),
        "kategori": q.query_per_dimensi(con, where, params, "kategori"),
        "metode_bayar": q.query_per_dimensi(con, where, params, "metode_bayar"),
        "pola_jam": q.query_pola_jam(con, where, params),
        "sampel": q.query_sampel(con, where, params, limit=1000),
    }


# DATA: hasil (DataFrame/dict) disimpan per kombinasi argumen. ttl = kedaluwarsa,
# max_entries = batas jumlah kombinasi filter yang disimpan (mencegah RAM membengkak).
hitung_dashboard_cached = st.cache_data(ttl="1h", max_entries=100, show_spinner=False)(hitung_dashboard)


# ---------------------------------------------------------------------------
# 2. Sidebar: filter dalam st.form (skrip baru dijalankan ulang saat tombol ditekan)
# ---------------------------------------------------------------------------
con = get_koneksi()
opsi = q.opsi_filter(con)

with st.sidebar:
    st.header("🔍 Filter")
    with st.form("form_filter"):
        wilayah = st.multiselect("Wilayah", opsi["wilayah"], placeholder="Semua wilayah")
        kategori = st.multiselect("Kategori", opsi["kategori"], placeholder="Semua kategori")
        rentang = st.date_input(
            "Rentang tanggal",
            value=(opsi["tgl_min"], opsi["tgl_max"]),
            min_value=opsi["tgl_min"], max_value=opsi["tgl_max"],
        )
        label_gran = st.radio("Granularitas tren", list(GRANULARITAS), index=1, horizontal=True)
        st.form_submit_button("Terapkan filter", type="primary", width="stretch")

    st.divider()
    tanpa_cache = st.toggle("Mode demo: matikan cache", help="Bandingkan waktu proses dengan dan tanpa @st.cache_data")
    if st.button("Kosongkan cache data", width="stretch"):
        st.cache_data.clear()

# date_input mengembalikan 1 tanggal saat pengguna baru memilih tanggal awal
if not isinstance(rentang, (tuple, list)) or len(rentang) != 2:
    st.info("Pilih tanggal awal dan akhir pada filter, lalu tekan **Terapkan filter**.")
    st.stop()
tgl_awal, tgl_akhir = rentang

# ---------------------------------------------------------------------------
# 3. Ambil data
# ---------------------------------------------------------------------------
fungsi = hitung_dashboard if tanpa_cache else hitung_dashboard_cached
t0 = time.perf_counter()
with st.spinner("Menjalankan kueri DuckDB ..."):
    d = fungsi(tuple(wilayah), tuple(kategori), tgl_awal, tgl_akhir, GRANULARITAS[label_gran])
durasi_ms = (time.perf_counter() - t0) * 1000

# ---------------------------------------------------------------------------
# 4. Tampilan
# ---------------------------------------------------------------------------
st.title("🛒 Dashboard Penjualan E-Commerce 2025")
st.caption(
    f"Sumber: {q.PATH_PARQUET.name} (2 juta transaksi sintetis) · "
    f"Periode {tgl_awal:%d %b %Y} – {tgl_akhir:%d %b %Y} · "
    f"Wilayah: {', '.join(wilayah) or 'semua'} · Kategori: {', '.join(kategori) or 'semua'}"
)

kpi, lalu = d["kpi"], d["kpi_lalu"]
if not kpi["jumlah_transaksi"]:
    st.warning("Tidak ada transaksi untuk kombinasi filter ini.")
    st.stop()


def delta(sekarang, sebelum):
    if not sebelum:
        return None
    return f"{(sekarang - sebelum) / sebelum:+.1%} vs periode sebelumnya"


k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah transaksi", f"{kpi['jumlah_transaksi']:,}".replace(",", "."),
          delta(kpi["jumlah_transaksi"], lalu["jumlah_transaksi"]), border=True)
k2.metric("Omzet", rupiah(kpi["omzet"]), delta(kpi["omzet"], lalu["omzet"]), border=True)
k3.metric("Rata-rata nilai transaksi", rupiah(kpi["rata_keranjang"]),
          delta(kpi["rata_keranjang"], lalu["rata_keranjang"]), border=True)
k4.metric("Rata-rata rating", f"{kpi['rata_rating']:.2f} ★".replace(".", ","),
          help=f"{kpi['tanpa_rating']:,} transaksi belum diberi rating (NULL, tidak dihitung)", border=True)

tab_tren, tab_dimensi, tab_waktu, tab_data = st.tabs(
    ["📈 Tren", "🗺️ Wilayah & Kategori", "🕒 Pola Waktu", "📋 Data"]
)

with tab_tren:
    metrik = st.radio("Metrik", ["omzet", "jumlah_transaksi"], horizontal=True,
                      format_func=lambda m: "Omzet" if m == "omzet" else "Jumlah transaksi")
    fig = px.line(d["tren"], x="periode", y=metrik, markers=label_gran != "Harian",
                  labels={"periode": "", "omzet": "Omzet (Rp)", "jumlah_transaksi": "Transaksi"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Lonjakan tanggal kembar (11.11, 12.12) paling jelas pada granularitas **Harian**.")

with tab_dimensi:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(d["wilayah"].sort("omzet"), x="omzet", y="wilayah", orientation="h",
                     title="Omzet per wilayah", labels={"omzet": "Omzet (Rp)", "wilayah": ""})
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.scatter(d["kategori"], x="jumlah_transaksi", y="rata_rating", size="omzet",
                         color="kategori", title="Kategori: volume vs rating (ukuran = omzet)",
                         labels={"jumlah_transaksi": "Transaksi", "rata_rating": "Rata-rata rating"})
        st.plotly_chart(fig, width="stretch")
    fig = px.bar(d["metode_bayar"], x="metode_bayar", y="jumlah_transaksi", color="rata_rating",
                 color_continuous_scale="RdYlGn", title="Metode pembayaran (warna = rata-rata rating)",
                 labels={"metode_bayar": "", "jumlah_transaksi": "Transaksi"})
    st.plotly_chart(fig, width="stretch")

with tab_waktu:
    pola = d["pola_jam"].with_columns(
        pl.col("hari").replace_strict(NAMA_HARI, return_dtype=pl.String).alias("nama_hari")
    )
    fig = px.density_heatmap(pola, x="jam", y="nama_hari", z="jumlah_transaksi", histfunc="sum",
                             nbinsx=24, category_orders={"nama_hari": list(NAMA_HARI.values())},
                             color_continuous_scale="Blues", title="Kapan pelanggan berbelanja?",
                             labels={"jam": "Jam", "nama_hari": "", "jumlah_transaksi": "Transaksi"})
    st.plotly_chart(fig, width="stretch")

with tab_data:
    st.write("1.000 transaksi dengan nilai terbesar (hasil `LIMIT` di DuckDB, bukan seluruh data).")
    st.dataframe(
        d["sampel"], hide_index=True, height=400,
        column_config={
            "waktu": st.column_config.DatetimeColumn("Waktu", format="DD MMM YYYY HH:mm"),
            "harga_satuan": st.column_config.NumberColumn("Harga satuan", format="localized"),
            "total": st.column_config.NumberColumn("Total", format="localized"),
            "rating": st.column_config.NumberColumn("Rating", format="%.1f ★"),
        },
    )
    st.download_button("⬇️ Unduh ringkasan per wilayah (CSV)", d["wilayah"].write_csv(),
                       file_name="ringkasan_wilayah.csv", mime="text/csv")

st.divider()
st.caption(
    f"⏱️ Waktu ambil data: **{durasi_ms:,.0f} ms** "
    f"({'tanpa cache, kueri dijalankan ulang' if tanpa_cache else 'dengan @st.cache_data'}). "
    "Coba ubah filter yang sama dua kali, lalu nyalakan mode demo."
)
