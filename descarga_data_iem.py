import requests
import pandas as pd
import time
from io import StringIO

# 1. Definir los aeropuertos y sus redes (networks) para el IEM.
# IEM clasifica por estado, así que necesitamos el mapeo: Network -> Station
airports_config = [
    {"network": "GA_ASOS", "station": "ATL"},
    {"network": "NY_ASOS", "station": "LGA"},
    {"network": "FL_ASOS", "station": "MCO"},
    {"network": "FL_ASOS", "station": "FLL"},
    {"network": "FL_ASOS", "station": "MIA"},
    {"network": "TX_ASOS", "station": "DFW"},
    {"network": "DC_ASOS", "station": "DCA"},
    {"network": "NJ_ASOS", "station": "EWR"},
    {"network": "FL_ASOS", "station": "TPA"},
    {"network": "IL_ASOS", "station": "ORD"},
    {"network": "PA_ASOS", "station": "PHL"},
    {"network": "CO_ASOS", "station": "DEN"},
    {"network": "CA_ASOS", "station": "LAX"},
    {"network": "MD_ASOS", "station": "BWI"},
    {"network": "NV_ASOS", "station": "LAS"},
    {"network": "MA_ASOS", "station": "BOS"}
]

# 2. Configuración de la API del IEM
BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Variables clave para el modelo predictivo aeronáutico:
# tmpc (Temp), dwpc (Dew Point), relh (Humidity), drct (Wind Dir), sknt (Wind Speed), 
# alti (Altimeter/Pressure), p01m (Precip 1hr), vsby (Visibility), gust (Wind Gusts), wxcodes (Weather Codes)
DATA_VARS = ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust", "wxcodes"]

all_weather_data = []


def parse_iem_csv(response_text: str) -> pd.DataFrame:
    """Parsea la respuesta CSV del IEM buscando el header real de columnas."""
    lines = response_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("station,valid"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("No se encontró el encabezado CSV en la respuesta del IEM.")

    csv_data = StringIO("\n".join(lines[header_idx:]))
    df = pd.read_csv(csv_data)
    df.columns = df.columns.str.strip()
    return df

print("Iniciando descarga de datos ASOS/METAR desde Iowa Mesonet para 2025...")

# 3. Bucle de descarga
for config in airports_config:
    airport = config["station"]
    network = config["network"]
    print(f"Descargando datos para {airport} ({network})...")
    
    # Construir parámetros
    base_params = [
        ("network", network),
        ("station", airport),
        ("year1", "2025"), ("month1", "1"), ("day1", "1"),
        ("year2", "2026"), ("month2", "1"), ("day2", "1"),  # Hasta fin de 2025
        ("tz", "Etc/UTC"),  # Todo en UTC para matchear luego
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "3"),  # Routine
        ("report_type", "4"),  # SPECI
    ]
    request_params = base_params + [("data", var) for var in DATA_VARS]

    try:
        response = requests.get(BASE_URL, params=request_params, timeout=60)
        response.raise_for_status() # Lanza excepción si hay error HTTP
        df_airport = parse_iem_csv(response.text)
        df_airport["source_station"] = airport
        df_airport["source_network"] = network
        
        all_weather_data.append(df_airport)
        print(f"  -> {len(df_airport)} registros descargados para {airport}.")
        
    except Exception as e:
        print(f"  [ERROR] Falló la descarga de {airport}: {e}")
    
    # Pausa de cortesía para el servidor universitario
    time.sleep(2)

# 4. Consolidación y Exportación
if all_weather_data:
    print("\nConsolidando los datos...")
    df_final = pd.concat(all_weather_data, ignore_index=True)
    
    filename = "clima_iem_asos_2025_utc.csv"
    df_final.to_csv(filename, index=False)
    print(f"¡Éxito! Archivo guardado como {filename} con {len(df_final)} filas.")
else:
    print("No se pudo descargar ningún dato.")