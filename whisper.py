# -*- coding: utf-8 -*-

!nvidia-smi
!pip install openai-whisper pydub -q
!apt update -qq && apt install -y ffmpeg -qq
print("✅ Instalación completada")

import whisper
import torch
import numpy as np
import pandas as pd
import os, glob, json, zipfile, time
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.effects import normalize
from google.colab import files

print(f"CUDA disponible: {torch.cuda.is_available()}")

CONFIG = {
    "model_size": "large",
    "language": "es",
    "no_speech_threshold": 0.3,
    "temperature": [0.0, 0.2, 0.4],  # Fallback automático si falla
    "condition_on_previous_text": False,
    "initial_prompt": (
        "Transcripción de llamada telefónica en español mexicano. "
        "Pueden haber dos personas hablando. "
        "Incluye todo el diálogo completo."
    ),
    # ── Parámetros de silencio ──────────────────────────────
    # Silencio mayor a este valor (ms) se elimina del audio
    # 2000ms = pausas naturales de conversación se conservan
    # Ajusta a 1500 si las llamadas tienen pausas muy cortas
    "min_silence_to_remove_ms": 2000,
    # dBFS para considerar silencio (-40 es estándar telefónico)
    # Baja a -50 si el audio tiene fondo ruidoso (música de espera)
    "silence_thresh_dbfs": -40,
    # Buffer que se conserva alrededor de cada fragmento de voz
    "keep_silence_ms": 300,
    "max_retries": 2,
}

print(f"Cargando modelo '{CONFIG['model_size']}'...")
model = whisper.load_model(CONFIG["model_size"])
print("✅ Modelo cargado")

def remove_all_long_silences(audio: AudioSegment) -> AudioSegment:
    """
    Detecta y elimina bloques de silencio LARGOS a lo largo de todo el audio.
    Conserva silencios cortos para mantener el ritmo natural del habla.

    Esto resuelve el caso: [mensaje pregrabado] [30s silencio] [conversación]
    → resultado:           [mensaje pregrabado] [conversación]
    """
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=CONFIG["min_silence_to_remove_ms"],
        silence_thresh=CONFIG["silence_thresh_dbfs"],
        seek_step=50,  # Mayor precisión (ms entre cada análisis)
    )

    if not nonsilent_ranges:
        print("    ⚠️ detect_nonsilent no encontró voz — usando audio original")
        return audio

    # Construir audio solo con los fragmentos que tienen voz
    # Se añade un buffer (keep_silence_ms) antes y después de cada fragmento
    segments = []
    total_removed_ms = 0

    for idx, (start, end) in enumerate(nonsilent_ranges):
        buffered_start = max(0, start - CONFIG["keep_silence_ms"])
        buffered_end   = min(len(audio), end + CONFIG["keep_silence_ms"])
        segments.append(audio[buffered_start:buffered_end])

        # Calcular cuánto silencio se eliminó entre fragmentos
        if idx > 0:
            prev_end = nonsilent_ranges[idx - 1][1] + CONFIG["keep_silence_ms"]
            gap = buffered_start - prev_end
            if gap > 0:
                total_removed_ms += gap

    cleaned = sum(segments)  # Concatenar todos los fragmentos de voz

    original_s = len(audio) / 1000
    cleaned_s  = len(cleaned) / 1000
    print(f"    ✂️  Silencio eliminado: {total_removed_ms/1000:.1f}s "
          f"| Audio: {original_s:.1f}s → {cleaned_s:.1f}s "
          f"| Fragmentos de voz encontrados: {len(nonsilent_ranges)}")

    return cleaned


def preprocess_audio(audio_path: str):
    """
    Pipeline completo de preprocesamiento:
    1. Cargar y convertir a mono 16kHz
    2. Normalizar volumen (crítico para audio telefónico bajo)
    3. Eliminar silencios largos en todo el audio
    4. Convertir a array float32 para Whisper
    """
    try:
        audio = AudioSegment.from_file(audio_path)

        # 1. Formato requerido por Whisper
        audio = audio.set_channels(1).set_frame_rate(16000)

        # 2. Normalizar — fundamental para llamadas con volumen bajo
        audio = normalize(audio)

        print(f"    📊 Original: {len(audio)/1000:.1f}s | dBFS: {audio.dBFS:.1f}")

        # 3. Eliminar silencios largos en TODO el audio
        audio = remove_all_long_silences(audio)

        # 4. Verificar que quede audio procesable (> 0.5s)
        if len(audio) < 500:
            print("    ⛔ Audio demasiado corto tras limpieza")
            return None, False

        # 5. Convertir a array float32 normalizado [-1.0, 1.0]
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(audio.array_type).max

        return samples, True

    except Exception as e:
        print(f"    ❌ Preprocesamiento falló: {e}")
        return None, False

def transcribe_single(audio_array: np.ndarray, attempt: int = 1) -> dict:
    return model.transcribe(
        audio_array,
        language=CONFIG["language"],
        no_speech_threshold=CONFIG["no_speech_threshold"],
        temperature=CONFIG["temperature"],
        condition_on_previous_text=CONFIG["condition_on_previous_text"],
        initial_prompt=CONFIG["initial_prompt"],
        verbose=False,
    )


def transcribe_with_retries(audio_path: str, audio_array: np.ndarray) -> tuple[str, dict, str]:
    """
    Intenta transcribir. Si el resultado es vacío, reintenta.
    Si todos los intentos fallan, activa modo fallback: transcribe
    el audio original SIN preprocesamiento (por si el trim fue excesivo).

    Retorna: (texto, result_dict, status)
    """
    result = None

    for attempt in range(1, CONFIG["max_retries"] + 2):
        try:
            print(f"    🔄 Intento {attempt}...")
            result = transcribe_single(audio_array, attempt)
            text = result["text"].strip()

            if text:
                print(f"    ✅ OK — {len(text)} caracteres | "
                      f"Segmentos: {len(result.get('segments', []))}")
                return text, result, "ok"
            else:
                print(f"    ⚠️ Vacío en intento {attempt}")

        except Exception as e:
            print(f"    ❌ Error intento {attempt}: {e}")

    # ── Fallback: audio sin preprocesar ──────────────────────
    # Si el silencio eliminado era en realidad señal útil (ej: música de espera
    # con voz encima), probamos con el audio crudo original
    print("    🆘 Fallback: transcribiendo audio sin preprocesamiento...")
    try:
        result_raw = model.transcribe(
            audio_path,
            language=CONFIG["language"],
            no_speech_threshold=0.1,   # Muy permisivo para el fallback
            temperature=[0.0, 0.2, 0.4, 0.6],
            condition_on_previous_text=False,
            initial_prompt=CONFIG["initial_prompt"],
            verbose=False,
        )
        text_raw = result_raw["text"].strip()
        if text_raw:
            print(f"    ✅ Fallback exitoso — {len(text_raw)} caracteres")
            return text_raw, result_raw, "ok_fallback"
    except Exception as e:
        print(f"    ❌ Fallback también falló: {e}")

    return "", result, "vacio"

def analyze_segments(result: dict) -> dict:
    segments = result.get("segments", []) if result else []
    if not segments:
        return {"num_segments": 0, "avg_no_speech_prob": None}
    probs = [s.get("no_speech_prob", 0) for s in segments]
    return {
        "num_segments": len(segments),
        "avg_no_speech_prob": round(np.mean(probs), 3),
        "max_no_speech_prob": round(max(probs), 3),
    }

def transcribe_audio_files(audio_folder: str, output_file: str = "transcripciones.csv",
                            already_processed: set = None):
    already_processed = already_processed or set()

    exts = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.aac', '*.ogg']
    audio_files = []
    for ext in exts:
        audio_files.extend(glob.glob(os.path.join(audio_folder, ext)))
        audio_files.extend(glob.glob(os.path.join(audio_folder, "**", ext), recursive=True))

    audio_files = list({f for f in audio_files if os.path.basename(f) not in already_processed})
    print(f"📁 Archivos a procesar: {len(audio_files)}")

    results, failed = [], []
    start_time = time.time()

    for i, audio_file in enumerate(audio_files):
        filename = os.path.basename(audio_file)
        print(f"\n[{i+1}/{len(audio_files)}] 📝 {filename}")

        audio_array, ok = preprocess_audio(audio_file)
        if not ok:
            failed.append({"archivo": filename, "motivo": "preprocesamiento_fallido"})
            continue

        text, result, status = transcribe_with_retries(audio_file, audio_array)
        seg_info = analyze_segments(result)

        if not text:
            failed.append({"archivo": filename, "motivo": "transcripcion_vacia", **seg_info})

        results.append({
            "archivo": filename,
            "transcripcion": text,
            "idioma": result.get("language", "error") if result else "error",
            "status": status,
            **seg_info,
            "procesado_en": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Guardado incremental
        if (i + 1) % 5 == 0:
            _save(results, output_file, failed, final=False)

    df = _save(results, output_file, failed, final=True)
    _summary(results, failed, start_time)
    return df


def _save(results, output_file, failed, final=False):
    if not results:
        return None
    prefix = "" if final else "progreso_"
    df = pd.DataFrame(results)
    df.to_csv(f"{prefix}{output_file}", index=False, encoding="utf-8-sig")
    json_out = f"{prefix}{output_file.replace('.csv', '.json')}"
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if failed:
        pd.DataFrame(failed).to_csv(f"fallos_{output_file}", index=False)
    return df


def _summary(results, failed, start_time):
    total = time.time() - start_time
    ok = sum(1 for r in results if "ok" in r["status"])
    print(f"\n{'='*45}")
    print(f"  Procesados : {len(results)}")
    print(f"  Exitosos   : {ok}")
    print(f"  Fallidos   : {len(failed)}")
    print(f"  Tiempo     : {total/60:.1f} min")
    print(f"{'='*45}")


def resume_transcription(audio_folder, progress_file="progreso_transcripciones.csv",
                          output_file="transcripciones.csv"):
    already = set()
    if os.path.exists(progress_file):
        df_prev = pd.read_csv(progress_file)
        already = set(df_prev[df_prev["status"].str.contains("ok", na=False)]["archivo"])
        print(f"📂 Omitiendo {len(already)} archivos ya exitosos")
    return transcribe_audio_files(audio_folder, output_file, already)

uploaded = files.upload()
AUDIO_FOLDER = "/content/audios"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

for filename in uploaded.keys():
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(filename, 'r') as z:
            z.extractall(AUDIO_FOLDER)
    else:
        os.rename(filename, os.path.join(AUDIO_FOLDER, filename))

df_results = transcribe_audio_files(AUDIO_FOLDER, "mis_transcripciones.csv")

if df_results is not None:
    print(df_results[["archivo", "status", "avg_no_speech_prob", "transcripcion"]].head(10))
    files.download("mis_transcripciones.csv")
    files.download("mis_transcripciones.json")