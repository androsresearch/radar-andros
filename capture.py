"""
Captura frames del radar de Bahamas (composite oficial del Bahamas Dept. of
Meteorology, servido por RainViewer) recortados a la zona de la Isla de Andros.

Guarda dos versiones por frame (cada 10 minutos):
  data/raw/    -> esquema "Black and White": el pixel codifica dBZ.
                  Decodificar: dBZ = R/2 - 32  (canal rojo del pixel)
                  Mismas unidades (dBZ) que la leyenda del website oficial.
  data/visual/ -> esquema a color para inspeccion visual rapida.

Fuente: RainViewer API (uso personal/educativo, atribucion requerida).
https://www.rainviewer.com/
"""

import io
import os

import requests
from datetime import datetime, timezone
from PIL import Image

API = "https://api.rainviewer.com/public/weather-maps.json"

# Tiles Web Mercator zoom 8 que cubren Andros completa (lat ~23.2-25.8 N)
Z = 8
X = 72
YS = (109, 110)

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

            tiles = []
            for y in YS:
                url = f"{host}{frame['path']}/512/{Z}/{X}/{y}/{color}/{opts}.png"
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                tiles.append(Image.open(io.BytesIO(r.content)).convert("RGBA"))

            # apilar los dos tiles verticalmente (norte arriba)
            combo = Image.new("RGBA", (512, 512 * len(tiles)))
            for i, tile in enumerate(tiles):
                combo.paste(tile, (0, 512 * i))

            os.makedirs(out_dir, exist_ok=True)
            combo.save(out, optimize=True)
            nuevos += 1
            print("guardado:", out)

    print(f"frames nuevos: {nuevos}")


if __name__ == "__main__":
    main()
