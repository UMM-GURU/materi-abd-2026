FROM quay.io/jupyter/datascience-notebook:latest

LABEL maintainer="Kuliah Analisis Big Data"
LABEL description="Docker environment untuk kuliah Analisis Big Data - Semester Ganjil 2026"

USER root

# ===================================
# System dependencies
# ===================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ===================================
# Python packages - Core Data Tools
# ===================================
RUN pip install --no-cache-dir \
    # DataFrame & SQL engines
    duckdb>=1.2.0 \
    "polars[pyarrow]>=1.20.0" \
    jupysql>=0.10.0 \
    # Visualization
    plotly>=6.0.0 \
    matplotlib>=3.9.0 \
    altair>=5.5.0 \
    # Dashboard
    streamlit>=1.45.0 \
    # Parallel computing
    "dask[complete]>=2025.1.0" \
    # Data access
    datasets>=3.0.0 \
    kaggle \
    # Machine Learning
    scikit-learn>=1.6.0 \
    # NLP
    transformers>=4.50.0 \
    # Geospatial
    geopandas>=1.0.0 \
    folium>=0.19.0 \
    # Data quality
    great-expectations>=1.3.0 \
    # Utilities
    pyarrow>=18.0.0 \
    openpyxl \
    requests \
    tqdm \
    wordcloud

# ===================================
# Jupyter extensions
# ===================================
RUN pip install --no-cache-dir \
    jupytext \
    ipywidgets

# ===================================
# Switch back to notebook user
# ===================================
USER ${NB_UID}

# ===================================
# Working directory
# ===================================
WORKDIR /home/jovyan/work

# ===================================
# Copy starter materials (if present)
# ===================================
COPY --chown=${NB_UID}:${NB_GID} notebooks/ /home/jovyan/work/notebooks/
COPY --chown=${NB_UID}:${NB_GID} data/ /home/jovyan/work/data/

# ===================================
# Expose ports: Jupyter (8888) + Streamlit (8501)
# ===================================
EXPOSE 8888 8501
