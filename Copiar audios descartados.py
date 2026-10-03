# -*- coding: utf-8 -*-

import os
import shutil
import pandas as pd
import json
import glob

# ==============================
# BASE: carpeta donde vive este script
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(BASE_DIR, "audios_descartados")

# posibles nombres de columna donde viene el audio
COLUMNAS_AUDIO = ["audio", "audio_path", "file", "filename"]

# extensiones de audio válidas
EXTENSIONES_AUDIO = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]


# ==============================
# UTILIDADES
# ==============================

def encontrar_columna_audio(df):
    for col in COLUMNAS_AUDIO:
        if col in df.columns:
            return col
    return None


def extraer_rutas_desde_csv(file_path):
    # ── Intento 1: leer con encabezado y buscar columna conocida ──────
    df = pd.read_csv(file_path, sep=None, engine="python", dtype=str)
    col_audio = encontrar_columna_audio(df)

    if col_audio:
        print(f"  📌 Columna de audio detectada: '{col_audio}'")
        return df[col_audio].dropna().tolist()

    # ── Intento 2: releer sin encabezado y escanear TODAS las celdas ──
    print("  📌 Sin columna reconocida — escaneando todas las celdas...")
    df = pd.read_csv(file_path, sep=None, engine="python", header=None, dtype=str)

    rutas = []
    ext = tuple(EXTENSIONES_AUDIO)
    for col in df.columns:
        for valor in df[col].dropna():
            v = str(valor).strip()
            if v.lower().endswith(ext):
                rutas.append(v)

    if not rutas:
        print(f"  ⚠️  No se encontraron rutas de audio. Columnas: {list(df.iloc[0])}")

    return rutas

    # ── Caso 3: nada reconocido ───────────────────────────────────────
    print(f"  ⚠️  No se encontró columna de audio. Columnas disponibles: {list(df.columns)}")
    return []


def extraer_rutas_desde_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rutas = []
    for item in data:
        for key in COLUMNAS_AUDIO:
            if key in item:
                rutas.append(item[key])
                break
    return rutas


def construir_indice_audios():
    """
    Recorre BASE_DIR completo y construye un diccionario
    { nombre_archivo: ruta_absoluta } para búsqueda rápida.
    Excluye la carpeta de salida para no procesar copias.
    """
    indice = {}
    for root, dirs, files in os.walk(BASE_DIR):
        # no entrar a la carpeta de salida
        dirs[:] = [d for d in dirs if os.path.join(root, d) != CARPETA_SALIDA]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in EXTENSIONES_AUDIO:
                indice[file] = os.path.join(root, file)
    return indice


# ==============================
# PROCESAMIENTO
# ==============================

def procesar_descartados():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Busca cualquier CSV o JSON en BASE_DIR (sin entrar a subcarpetas)
    archivos_csv  = glob.glob(os.path.join(BASE_DIR, "*.csv"))
    archivos_json = glob.glob(os.path.join(BASE_DIR, "*.json"))
    archivos = archivos_csv + archivos_json

    if not archivos:
        print("❌ No se encontraron archivos .csv ni .json en la carpeta del script.")
        return

    print(f"📂 Archivos encontrados: {len(archivos)}")

    # Construir índice de audios una sola vez
    print("🔍 Indexando audios disponibles...")
    indice_audios = construir_indice_audios()
    print(f"   {len(indice_audios)} audio(s) indexados.\n")

    total_encontrados    = 0
    total_no_encontrados = 0
    ya_copiados = set()  # evitar duplicados si el mismo audio aparece en varios CSVs

    for file in archivos:
        print(f"📄 Procesando: {os.path.basename(file)}")

        if file.endswith(".csv"):
            rutas = extraer_rutas_desde_csv(file)
        else:
            rutas = extraer_rutas_desde_json(file)

        print(f"   🔎 Entradas a procesar: {len(rutas)}")

        for ruta in rutas:
            nombre_audio = os.path.basename(str(ruta).strip())

            # validar extensión
            if not any(nombre_audio.lower().endswith(ext) for ext in EXTENSIONES_AUDIO):
                continue

            if nombre_audio in ya_copiados:
                continue

            if nombre_audio in indice_audios:
                origen  = indice_audios[nombre_audio]
                destino = os.path.join(CARPETA_SALIDA, nombre_audio)
                shutil.copy2(origen, destino)
                ya_copiados.add(nombre_audio)
                total_encontrados += 1
            else:
                print(f"   ⚠️  No encontrado: {nombre_audio}")
                total_no_encontrados += 1

        print()

    print("==============================")
    print(f"✅ Audios copiados : {total_encontrados}")
    print(f"❌ No encontrados  : {total_no_encontrados}")
    print(f"📁 Destino         : {CARPETA_SALIDA}")
    print("==============================")


# ==============================
# EJECUCIÓN
# ==============================

if __name__ == "__main__":
    procesar_descartados()