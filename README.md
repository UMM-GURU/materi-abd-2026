# Kuliah Analisis Big Data — Docker Environment

## Quick Start

### Prasyarat
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) terinstal
- Minimal 8GB RAM tersedia untuk Docker

### Menjalankan JupyterLab

```bash
# 1. Clone atau download folder ini
# 2. Masuk ke folder
cd bigdata-kuliah

# 3. Build dan jalankan (jalankan pertama kali, atau setelah Dockerfile berubah)
docker compose up --build

# 4. Buka di browser:
#    http://localhost:8888/lab
```

Untuk penggunaan lokal, proyek ini menonaktifkan token dan password Jupyter agar
mahasiswa tidak perlu melakukan login. Biarkan terminal tersebut tetap berjalan.
Setelah image pernah dibangun, perintah berikut cukup digunakan untuk menjalankan
kembali container:

```bash
docker compose up
```

> **Catatan (mulai Pertemuan 9):** Dockerfile telah diperbarui dengan paket
> `lightgbm` dan `joblib`. Jika image sudah pernah dibangun sebelumnya, jalankan
> ulang dengan `docker compose up --build` agar paket baru terinstal.

### Menjalankan file Pertemuan 1

File `notebooks/P01_pengantar/01_pengantar_bigdata.py` sengaja disimpan sebagai file
Python dengan format cell `# %%`. Untuk pengalaman paling mudah di JupyterLab, buka
file notebook native `01_pengantar_bigdata.ipynb` yang sudah tersedia di folder yang
sama.

Untuk menjalankan seluruh file dari terminal:

```bash
docker compose exec jupyter \
    python /home/jovyan/work/notebooks/P01_pengantar/01_pengantar_bigdata.py
```

Di JupyterLab, buka folder `notebooks/P01_pengantar`, lalu buka
`01_pengantar_bigdata.ipynb` dan jalankan cell satu per satu. File `.py` tetap
disimpan sebagai sumber versi teks dengan format `# %%`.

### Alternatif: menjalankan cell `.py` dari VS Code

Jika VS Code menampilkan **Select a kernel to run**, hubungkan VS Code ke Jupyter
server yang berjalan di Docker:

1. Jalankan container dengan `docker compose up`.
2. Di VS Code, buka Command Palette (`Cmd+Shift+P`) lalu pilih **Jupyter: Specify
    local or remote Jupyter server for connections**.
3. Masukkan URL server `http://localhost:8888/`.
4. Buka `01_pengantar_bigdata.py`, lalu klik **Select Kernel** dan pilih kernel dari
    Jupyter server Docker.

File `.py` dan `.ipynb` berisi materi yang sama. Jika materi `.py` diubah, perbarui
notebook dengan perintah berikut:

```bash
docker compose exec jupyter jupytext \
    --to notebook \
    /home/jovyan/work/notebooks/P01_pengantar/01_pengantar_bigdata.py \
    -o /home/jovyan/work/notebooks/P01_pengantar/01_pengantar_bigdata.ipynb
```

### Port yang digunakan
| Port | Layanan |
|------|---------|
| 8888 | JupyterLab |
| 8501 | Streamlit |
| 8787 | Dask Dashboard |

### Struktur Folder

```
bigdata-kuliah/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── notebooks/          ← Taruh notebook di sini
│   ├── P01_pengantar/
│   ├── P02_polars/
│   └── ...
├── data/               ← Taruh dataset di sini
│   ├── raw/
│   └── processed/
├── tugas1-template/    ← Template Tugas 1 (lihat README di dalamnya)
└── tugas2-template/    ← Template Tugas 2 (lihat README di dalamnya)
```

### Menjalankan Streamlit dari dalam Container

Streamlit sudah terpasang di image Docker. Perintah `streamlit` **hanya ada di dalam container**, tidak di terminal laptop (Mac/Windows), sehingga ada dua cara menjalankannya.

**Cara 1: Terminal JupyterLab** (buka http://localhost:8888 → *File → New → Terminal*):
```bash
cd ~/work/notebooks/P10_streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

**Cara 2: Terminal laptop** (dari folder `bigdata-kuliah`, container harus sudah berjalan):
```bash
docker compose exec -w /home/jovyan/work/notebooks/P10_streamlit jupyter \
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Buka **http://localhost:8501** di browser, lalu hentikan dengan `Ctrl+C`. Opsi `--server.address 0.0.0.0` wajib agar browser laptop dapat mengakses Streamlit di dalam container.

### Troubleshooting

**Port sudah terpakai?**
```bash
docker compose down
# Edit docker-compose.yml, ganti port mapping
docker compose up
```

**Perlu rebuild setelah update Dockerfile?**
```bash
docker compose up --build --force-recreate
```
