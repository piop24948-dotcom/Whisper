# -*- coding: utf-8 -*-

import pandas as pd
import json
import re
import os
import glob

# ==============================
# CONFIGURACIÓN
# ==============================
MIN_WORDS = 1  # Mínimo de palabras DESPUÉS de quitar el disclaimer

FRASES_DESCARTAR = [
    "les informamos que por cuestiones de calidad, esta llamada puede ser grabada",
    "les informamos que por cuestiones de calidad esta llamada puede ser grabada",
    "les informamos que por cuestiones de calidad",
    "Pueden haber dos personas hablando. Incluye todo el diálogo completo.",
    "¡Suscríbete!",
    "¡Gracias por ver el video!",
    "Pueden haber dos personas hablando. Incluye todo el diálogo completo. SILENCIO",
    "Les informamos que por cuestiones de",
    "¡Suscríbete al canal!",
    "Gracias por ver el video",
]

# ==============================
# NORMALIZACIÓN
# ==============================

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""

    texto = texto.strip()

    # Corrige problemas típicos de encoding (Ã³, Â¿, etc.)
    try:
        texto = texto.encode('latin1').decode('utf-8')
    except:
        pass

    return texto


# ==============================
# LIMPIEZA DEL DISCLAIMER
# ==============================

def quitar_disclaimer(texto):
    """
    Elimina el mensaje pregrabado del IVR del texto.
    Retorna el texto limpio sin el disclaimer.
    """
    texto_limpio = texto.lower()

    # Ordenar frases de más larga a más corta para evitar reemplazos parciales
    frases_ordenadas = sorted(FRASES_DESCARTAR, key=len, reverse=True)

    for frase in frases_ordenadas:
        texto_limpio = texto_limpio.replace(frase, " ")

    # Limpiar espacios múltiples que quedaron
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()

    return texto_limpio


# ==============================
# VALIDACIÓN PRINCIPAL
# ==============================

def es_conversacion_valida(texto):
    """
    Descarta ÚNICAMENTE:
    1. Textos vacíos o nulos
    2. Textos que solo contienen el disclaimer (sin conversación real)
    
    Todo lo demás se conserva.
    """
    if not isinstance(texto, str):
        return False

    texto_normalizado = normalizar_texto(texto)

    # 1. Vacío después de normalizar
    if texto_normalizado.strip() == "":
        return False

    # 2. Quitar el disclaimer y ver qué queda
    texto_sin_disclaimer = quitar_disclaimer(texto_normalizado)

    # 3. Si no quedan palabras reales tras quitar el disclaimer → descartar
    palabras_restantes = [p for p in texto_sin_disclaimer.split() if len(p) > 1]

    if len(palabras_restantes) < MIN_WORDS:
        return False

    return True


# ==============================
# CSV
# ==============================

def limpiar_csv(file_path):
    print(f"\n📄 Procesando CSV: {file_path}")

    df = pd.read_csv(file_path)
    total = len(df)

    df["valido"] = df["transcripcion"].apply(es_conversacion_valida)

    df_limpio = df[df["valido"] == True].drop(columns=["valido"])
    df_descartado = df[df["valido"] == False].drop(columns=["valido"])

    output_limpio = file_path.replace(".csv", "_limpio.csv")
    output_descartado = file_path.replace(".csv", "_descartados.csv")

    df_limpio.to_csv(output_limpio, index=False)
    df_descartado.to_csv(output_descartado, index=False)

    print(f"📊 Total:       {total}")
    print(f"✅ Conservados: {len(df_limpio)}")
    print(f"🗑️  Eliminados:  {len(df_descartado)}")
    print(f"💾 Limpio:      {output_limpio}")
    print(f"💾 Descartados: {output_descartado}")


# ==============================
# JSON
# ==============================

def limpiar_json(file_path):
    print(f"\n📄 Procesando JSON: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    data_limpia = []
    data_descartada = []

    for item in data:
        texto = item.get("transcripcion", "")

        if es_conversacion_valida(texto):
            data_limpia.append(item)
        else:
            data_descartada.append(item)

    output_limpio = file_path.replace(".json", "_limpio.json")
    output_descartado = file_path.replace(".json", "_descartados.json")

    with open(output_limpio, "w", encoding="utf-8") as f:
        json.dump(data_limpia, f, ensure_ascii=False, indent=2)

    with open(output_descartado, "w", encoding="utf-8") as f:
        json.dump(data_descartada, f, ensure_ascii=False, indent=2)

    print(f"📊 Total:       {total}")
    print(f"✅ Conservados: {len(data_limpia)}")
    print(f"🗑️  Eliminados:  {len(data_descartada)}")
    print(f"💾 Limpio:      {output_limpio}")
    print(f"💾 Descartados: {output_descartado}")


# ==============================
# DETECCIÓN AUTOMÁTICA
# ==============================

def procesar_carpeta():
    current_path = os.getcwd()

    csv_files = glob.glob(os.path.join(current_path, "*.csv"))
    json_files = glob.glob(os.path.join(current_path, "*.json"))

    if not csv_files and not json_files:
        print("❌ No se encontraron archivos CSV o JSON en la carpeta")
        return

    print(f"📂 Carpeta actual: {current_path}")
    print(f"📄 CSV encontrados:  {len(csv_files)}")
    print(f"📄 JSON encontrados: {len(json_files)}")

    for file in csv_files:
        if "_limpio" not in file and "_descartados" not in file:
            limpiar_csv(file)

    for file in json_files:
        if "_limpio" not in file and "_descartados" not in file:
            limpiar_json(file)


# ==============================
# EJECUCIÓN
# ==============================

if __name__ == "__main__":
    procesar_carpeta()