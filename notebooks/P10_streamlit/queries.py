"""Lapisan kueri (backend) dashboard Pertemuan 10.

Semua kerja berat (filter, agregasi) dilakukan DuckDB langsung di file Parquet.
Streamlit hanya menerima hasil ringkas (puluhan s.d. ribuan baris), bukan 2 juta baris mentah.

Modul ini sengaja tidak meng-import streamlit sehingga bisa diuji di notebook.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

DATA_DIR = Path(__file__).resolve().parent / "data"
PATH_PARQUET = DATA_DIR / "transaksi_2025.parquet"
N_BARIS_DEFAULT = 2_000_000

WILAYAH = ["DKI Jakarta", "Jawa Barat", "Jawa Tengah", "Jawa Timur", "Banten",
           "Sumatera Utara", "Sumatera Selatan", "Bali", "Kalimantan Timur", "Sulawesi Selatan"]
KATEGORI = ["Elektronik", "Fashion", "Kecantikan", "Rumah Tangga", "Olahraga", "Makanan"]
METODE_BAYAR = ["E-Wallet", "Transfer Bank", "COD", "Kartu Kredit", "Paylater"]


# ---------------------------------------------------------------------------
# 1. Pembuatan dataset sintetis (sekali saja)
# ---------------------------------------------------------------------------
def siapkan_data(path: Path = PATH_PARQUET, n_baris: int = N_BARIS_DEFAULT) -> Path:
    """Membangkitkan transaksi e-commerce sintetis 2025 langsung ke Parquet dengan DuckDB.

    setseed() + 1 thread membuat hasil random() selalu sama (deterministik).
    Data diurutkan berdasarkan waktu agar filter tanggal bisa melompati row group.
    """
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    def pilih(daftar: list[str], bobot: list[float], u: str) -> str:
        """CASE ... WHEN u < batas THEN nilai: memilih kategori dengan bobot tertentu."""
        batas, kumulatif, cabang = sum(bobot), 0.0, []
        for nilai, w in zip(daftar[:-1], bobot[:-1]):
            kumulatif += w / batas
            cabang.append(f"WHEN {u} < {kumulatif:.4f} THEN '{nilai}'")
        return f"CASE {' '.join(cabang)} ELSE '{daftar[-1]}' END"

    sql = f"""
    COPY (
        WITH acak AS (
            SELECT i, random() AS u1, random() AS u2, random() AS u3, random() AS u4,
                   random() AS u5, random() AS u6, random() AS u7, random() AS u8, random() AS u9
            FROM range({n_baris}) t(i)
        ), dasar AS (
            SELECT *,
                {pilih(WILAYAH, [24, 14, 11, 10, 8, 8, 7, 6, 6, 6], "u1")} AS wilayah,
                {pilih(KATEGORI, [12, 27, 19, 17, 10, 15], "u2")} AS kategori,
                {pilih(METODE_BAYAR, [38, 22, 16, 10, 14], "u3")} AS metode_bayar,
                -- Tren naik sepanjang tahun (pangkat < 1 memadatkan ke akhir tahun)
                DATE '2025-01-01' + floor(pow(u4, 0.8) * 365)::INT AS tgl0
            FROM acak
        ), waktu AS (
            SELECT *,
                CASE WHEN u5 < 0.05 THEN make_date(2025, 1 + floor(u9 * 12)::INT, 1 + floor(u9 * 12)::INT)
                     WHEN u5 < 0.17 THEN least(tgl0 + (6 - least(isodow(tgl0), 6))::INT, DATE '2025-12-31') -- geser ke akhir pekan
                     ELSE tgl0 END AS tgl,
                -- Jam belanja: puncak malam hari (19-22)
                CASE WHEN u6 < 0.40 THEN 19 + floor(u6 / 0.40 * 4)::INT
                     ELSE floor((u6 - 0.40) / 0.60 * 24)::INT END AS jam
            FROM dasar
        )
        SELECT
            row_number() OVER (ORDER BY tgl, jam, i) AS transaksi_id,
            tgl + to_hours(jam) + to_minutes(floor(u9 * 60)::INT) AS waktu,
            wilayah, kategori, metode_bayar,
            1 + floor(pow(u7, 3) * 5)::INT AS jumlah,
            round(CASE kategori
                    WHEN 'Elektronik'   THEN 150000 + pow(u8, 2) * 4850000
                    WHEN 'Fashion'      THEN  50000 + pow(u8, 2) *  450000
                    WHEN 'Kecantikan'   THEN  25000 + pow(u8, 2) *  300000
                    WHEN 'Rumah Tangga' THEN  30000 + pow(u8, 2) *  700000
                    WHEN 'Olahraga'     THEN  60000 + pow(u8, 2) * 1200000
                    ELSE                       10000 + pow(u8, 2) *  140000 END, -2) AS harga_satuan,
            -- 8% transaksi belum diberi rating (NULL); Elektronik & COD cenderung lebih rendah
            CASE WHEN u1 * 1000 % 1 < 0.08 THEN NULL
                 ELSE round(least(5.0, greatest(1.0,
                      5.0 - CASE kategori WHEN 'Elektronik' THEN 0.4 WHEN 'Fashion' THEN 0.2 ELSE 0.0 END
                          - CASE metode_bayar WHEN 'COD' THEN 0.25 ELSE 0.0 END
                          - pow(u3 * 1000 % 1, 2) * 2.0)), 1) END AS rating
        FROM waktu
        ORDER BY waktu
    ) TO '{path.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 122880);
    """
    con = duckdb.connect()
    con.execute("SET threads = 1")
    con.execute("SET enable_progress_bar = false")
    con.execute("SELECT setseed(0.2026)")
    con.execute(sql)
    con.close()
    return path
    path.parent.mkdir(parents=True, exist_ok=True)

    def acak(k: int) -> str:  # bilangan acak semu 0..1 yang deterministik
        return f"((hash(i * {k} + {k * 7919}) % 1000000) / 1000000.0)"

    wil = ", ".join(f"'{w}'" for w in WILAYAH)
    kat = ", ".join(f"'{k}'" for k in KATEGORI)
    bayar = ", ".join(f"'{m}'" for m in METODE_BAYAR)
    sql = f"""
    COPY (
        WITH dasar AS (
            SELECT
                i,
                -- Wilayah: Jawa lebih padat (distribusi miring via pangkat)
                [{wil}][1 + floor(pow({acak(1)}, 1.6) * {len(WILAYAH)})::INT] AS wilayah,
                [{kat}][1 + floor({acak(2)} * {len(KATEGORI)})::INT]          AS kategori,
                [{bayar}][1 + floor(pow({acak(3)}, 1.3) * {len(METODE_BAYAR)})::INT] AS metode_bayar,
                {acak(4)} AS u_hari, {acak(5)} AS u_jam, {acak(6)} AS u_harga,
                {acak(7)} AS u_qty,  {acak(8)} AS u_rating, {acak(9)} AS u_null
            FROM range({n_baris}) t(i)
        ), waktu AS (
            SELECT *,
                -- Hari dalam setahun; tanggal kembar & akhir tahun lebih ramai
                CASE WHEN u_hari < 0.06 THEN make_date(2025, 1 + (i % 12)::INT, 1 + (i % 12)::INT)
                     ELSE DATE '2025-01-01' + floor(pow(u_hari, 0.9) * 365)::INT END AS tgl,
                -- Jam belanja: puncak malam hari (19-22)
                CASE WHEN u_jam < 0.45 THEN 19 + floor(u_jam / 0.45 * 4)::INT
                     ELSE floor((u_jam - 0.45) / 0.55 * 24)::INT END AS jam
            FROM dasar
        )
        SELECT
            i + 1 AS transaksi_id,
            tgl + to_hours(jam) + to_minutes((i % 60)::INT) AS waktu,
            wilayah, kategori, metode_bayar,
            1 + floor(pow(u_qty, 3) * 5)::INT AS jumlah,
            round(CASE kategori
                    WHEN 'Elektronik'   THEN 150000 + u_harga * 4850000
                    WHEN 'Fashion'      THEN  50000 + u_harga *  450000
                    WHEN 'Kecantikan'   THEN  25000 + u_harga *  300000
                    WHEN 'Rumah Tangga' THEN  30000 + u_harga *  700000
                    WHEN 'Olahraga'     THEN  60000 + u_harga * 1200000
                    ELSE                       10000 + u_harga *  140000 END, -2) AS harga_satuan,
            -- 8% transaksi belum diberi rating (NULL)
            CASE WHEN u_null < 0.08 THEN NULL
                 ELSE round(least(5.0, 2.5 + pow(u_rating, 0.5) * 2.6), 1) END AS rating
        FROM waktu
    ) TO '{path.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 122880);
    """
    con = duckdb.connect()
    con.execute(sql)
    con.close()
    return path


# ---------------------------------------------------------------------------
# 2. Koneksi
# ---------------------------------------------------------------------------
def buat_koneksi(path: Path = PATH_PARQUET) -> duckdb.DuckDBPyConnection:
    """Koneksi DuckDB in-memory dengan VIEW `transaksi` di atas file Parquet.

    VIEW tidak menyalin data: setiap kueri membaca Parquet langsung (hanya kolom
    dan row group yang dibutuhkan).
    """
    siapkan_data(path)
    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE VIEW transaksi AS
        SELECT *, jumlah * harga_satuan AS total, CAST(waktu AS DATE) AS tanggal
        FROM read_parquet('{Path(path).as_posix()}')
    """)
    return con


def _jalankan(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pl.DataFrame:
    # cursor() = koneksi turunan yang aman dipakai di thread berbeda (setiap sesi Streamlit = 1 thread)
    cur = con.cursor()
    try:
        return cur.sql(sql, params=params or None).pl()
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# 3. Filter sebagai klausa WHERE berparameter (aman dari SQL injection)
# ---------------------------------------------------------------------------
def bangun_filter(wilayah=None, kategori=None, tgl_awal: date | None = None,
                  tgl_akhir: date | None = None) -> tuple[str, list]:
    syarat, params = [], []
    if wilayah:
        syarat.append(f"wilayah IN ({', '.join('?' * len(wilayah))})")
        params += list(wilayah)
    if kategori:
        syarat.append(f"kategori IN ({', '.join('?' * len(kategori))})")
        params += list(kategori)
    if tgl_awal:
        syarat.append("tanggal >= ?")
        params.append(tgl_awal)
    if tgl_akhir:
        syarat.append("tanggal <= ?")
        params.append(tgl_akhir)
    where = ("WHERE " + " AND ".join(syarat)) if syarat else ""
    return where, params


# ---------------------------------------------------------------------------
# 4. Kueri dashboard (semuanya mengembalikan hasil kecil)
# ---------------------------------------------------------------------------
def opsi_filter(con) -> dict:
    df = _jalankan(con, """
        SELECT list(DISTINCT wilayah ORDER BY wilayah) AS wilayah,
               list(DISTINCT kategori ORDER BY kategori) AS kategori,
               min(tanggal) AS tgl_min, max(tanggal) AS tgl_max
        FROM transaksi
    """)
    return df.row(0, named=True)


def query_kpi(con, where: str, params: list) -> dict:
    return _jalankan(con, f"""
        SELECT count(*)                 AS jumlah_transaksi,
               sum(total)               AS omzet,
               avg(total)               AS rata_keranjang,
               avg(rating)              AS rata_rating,
               count(*) - count(rating) AS tanpa_rating
        FROM transaksi {where}
    """, params).row(0, named=True)


def query_tren(con, where: str, params: list, granularitas: str = "week") -> pl.DataFrame:
    if granularitas not in {"day", "week", "month"}:
        raise ValueError("granularitas harus day/week/month")
    return _jalankan(con, f"""
        SELECT date_trunc('{granularitas}', tanggal) AS periode,
               count(*) AS jumlah_transaksi, sum(total) AS omzet
        FROM transaksi {where}
        GROUP BY 1 ORDER BY 1
    """, params)


def query_per_dimensi(con, where: str, params: list, dimensi: str) -> pl.DataFrame:
    if dimensi not in {"wilayah", "kategori", "metode_bayar"}:
        raise ValueError("dimensi tidak dikenal")
    return _jalankan(con, f"""
        SELECT {dimensi}, count(*) AS jumlah_transaksi, sum(total) AS omzet,
               round(avg(rating), 2) AS rata_rating
        FROM transaksi {where}
        GROUP BY 1 ORDER BY omzet DESC
    """, params)


def query_pola_jam(con, where: str, params: list) -> pl.DataFrame:
    return _jalankan(con, f"""
        SELECT isodow(waktu) AS hari, hour(waktu) AS jam, count(*) AS jumlah_transaksi
        FROM transaksi {where}
        GROUP BY 1, 2 ORDER BY 1, 2
    """, params)


def query_sampel(con, where: str, params: list, limit: int = 1000) -> pl.DataFrame:
    return _jalankan(con, f"""
        SELECT transaksi_id, waktu, wilayah, kategori, metode_bayar,
               jumlah, harga_satuan, total, rating
        FROM transaksi {where}
        ORDER BY total DESC
        LIMIT {int(limit)}
    """, params)


def ukur(fungsi, *args, **kwargs):
    """Menjalankan fungsi dan mengembalikan (hasil, detik)."""
    t0 = time.perf_counter()
    hasil = fungsi(*args, **kwargs)
    return hasil, time.perf_counter() - t0
