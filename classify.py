import sys
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt
import json

# Verifica che sia stato passato un file
if len(sys.argv) < 2:
    print("Specifica il percorso del file audio da classificare.")
    sys.exit(1)

audio_path = sys.argv[1]

# Carica il modello
model = load_model("/home/ric/ProgettoIoT/modello.h5")

def estrai_mfcc(path_audio):
    y, sr = librosa.load(path_audio, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Assicura che abbia 174 frame
    max_len = 174
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    mfcc = mfcc.reshape((40, 174, 1))  # formato immagine CNN
    return np.expand_dims(mfcc, axis=0)  # shape: (1, 40, 174, 1)

# Estrai caratteristiche
X = estrai_mfcc(audio_path)

print("Forma input:", X.shape)

# Predizione
pred = model.predict(X)
classe = np.argmax(pred)
print("🎯 Classe predetta:", classe)

# --- Integrazione MQTT ---
BROKER = "localhost"     # Cambia se necessario
PORT = 1883
TOPIC = "progetto_iot/riconoscimento"

# Setup client MQTT
client = mqtt.Client()
client.connect(BROKER, PORT)

# Crea messaggio JSON con risultati
message = {
    "file": audio_path,
    "label": int(classe),
    "confidence": float(np.max(pred))
}
message_json = json.dumps(message)

# Pubblica messaggio
client.publish(TOPIC, message_json)
print(f"Messaggio MQTT pubblicato su '{TOPIC}': {message_json}")

client.disconnect()


# --- Integrazione InfluxDB ---
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

# Parametri di connessione (modifica se necessario)
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "MTRLrR3BhYJIzTkOO7TN9iO2Rtk9GKGJBuhsp65lA9hpvoKgqt0wL6ZM81L3BlZC3Uu49yiGiFfwDo54NZ7kFA=="
INFLUX_ORG = "ric"
INFLUX_BUCKET = "progetto_iot"

try:
    influx_client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)

    point = (
        Point("audio_recognition")
        .tag("file", audio_path)
        .field("label", int(classe))
        .field("confidence", float(np.max(pred)))
        .time(datetime.utcnow(), WritePrecision.NS)
    )

    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
    print("✅ Dati scritti in InfluxDB.")

except Exception as e:
    print(f"❌ Errore scrittura InfluxDB: {e}")
