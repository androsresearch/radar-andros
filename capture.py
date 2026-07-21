"""
Captura frames del radar de Bahamas (composite oficial del Bahamas Dept. of
Meteorology, servido por RainViewer) centrados en la Isla de Andros.

Guarda dos versiones por frame (cada 10 minutos):
  data/raw/    -> esquema "Black and White": el pixel codifica dBZ.
                  Decodificar: dBZ = R/2 - 32  (canal rojo del pixel)
                  Mismas unidades (dBZ) que la leyenda del website oficial.
  data/visual/ -> esquema a color para inspeccion visual rapida.

Fuente: RainViewer API (uso personal/educativo, atribucion requerida).
https://www.rainviewer.com/
"""

import os

import requests
from datetime import datetime, timezone

API = "https://api.rainviewer.com/public/weather-maps.json"

# Centro de la Isla de Andros. Zoom 7 es el maximo que permite la API.
# Imagen 512px a zoom 7 cubre ~2.8 grados: Andros completa con margen.
LAT, LON = 24.45, -78.0
ZOOM = 7
SIZE = 512

# nombre -> (esquema de color, opciones smooth_snow)
# raw: color 0 (dBZ en escala de grises), sin suavizado (0_0) para no alterar valores
# visual: color 4 (Universal Blue), suavizado (1_1) solo para mirar
SCHEMES = {"raw": ("0", "0_0"), "visual": ("4", "1_1")}


def main() -> None:
    meta = requests.get(API, timeout=30).json()
    host = meta["host"]
    nuevos = 0

    for frame in meta["radar"]["past"]:
        ts = frame["time"]
        dt = datetime.fromtimestamp(ts, timezone.utc)

        for kind, (color, opts) in SCHEMES.items():
            out_dir = os.path.join("data", kind, dt.strftime("%Y/%m/%d"))
            out = os.path.join(out_dir, dt.strftime(f"andros_%Y%m%d_%H%MZ_{kind}.png"))
            if os.path.exists(out):
                continue

            url = f"{host}{frame['path']}/{SIZE}/{ZOOM}/{LAT}/{LON}/{color}/{opts}.png"
            r = requests.get(url, timeout=60)
            r.raise_for_status()

            os.makedirs(out_dir, exist_ok=True)
            with open(out, "wb") as f:
                f.write(r.content)
            nuevos += 1
            print("guardado:", out)

    print(f"frames nuevos: {nuevos}")


if __name__ == "__main__":
    main()
