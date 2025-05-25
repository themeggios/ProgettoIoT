import sys
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import datetime

# Verifica che sia stato passato un file
if len(sys.argv) < 2:
    print("❌ Errore: specifica il percorso del file audio da classificare.")
    sys.exit(1)

audio_path = sys.argv[1]

# Carica il modello
model = load_model("modello.h5")

def estrai_mfcc(path_audio):
    y, sr = librosa.load(path_audio, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    max_len = 174
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    mfcc = mfcc.reshape((40, 174, 1))
    return np.expand_dims(mfcc, axis=0)

# Estrai caratteristiche
X = estrai_mfcc(audio_path)

print("Forma input:", X.shape)

# Predizione
pred = model.predict(X)
classe = int(np.argmax(pred))
print("🎯 Classe predetta:", classe)

# ========== InfluxDB ==========

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "MTRLrR3BhYJIzTkOO7TN9iO2Rtk9GKGJBuhsp65lA9hpvoKgqt0wL6ZM81L3BlZC3Uu49yiGiFfwDo54NZ7kFA=="
INFLUX_ORG = "ric"
INFLUX_BUCKET = "audioDB"

client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

write_api = client.write_api(write_options=SYNCHRONOUS)

point = Point("audio_classification") \
    .tag("modello", "modello.h5") \
    .field("classe", classe) \
    .field("file", audio_path) \
    .time(datetime.datetime.utcnow())

write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
print("✅ Risultato salvato su InfluxDB.")
