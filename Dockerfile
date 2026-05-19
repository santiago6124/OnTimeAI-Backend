FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para LightGBM y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (cache layer)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código fuente y módulos
COPY api.py predict.py ./
COPY ontimeai/ ./ontimeai/
COPY feature_engineering_v7/ ./feature_engineering_v7/

# Artefactos del modelo activo y de soporte
COPY artifacts/4year_v9/ ./artifacts/4year_v9/
COPY artifacts/4year_v9_recal/ ./artifacts/4year_v9_recal/
COPY artifacts/lineage_fallback.joblib ./artifacts/
COPY artifacts/airport_pagerank.json ./artifacts/
COPY artifacts/distance_lookup.csv ./artifacts/
COPY artifacts/tail_to_aircraft_family.json ./artifacts/

# Universo de aeropuertos y base de datos live
COPY airports_universe.csv .
COPY live_data.db .

ENV ACTIVE_MODEL=4year_v9
ENV PORT=8080

EXPOSE 8080

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
