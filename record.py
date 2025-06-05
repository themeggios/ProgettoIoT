#!/usr/bin/env python
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

# === Parametri ===
DURATION = 3  # durata registrazione in secondi
SAMPLERATE = 22050  # frequenza di campionamento
CHANNELS = 1  # mono
DEVICE_ID = 2  # microfono della webcam

# === Nome file con timestamp ===
filename = "/home/ric/ProgettoIoT/recording.wav"

print("Inizio registrazione...")
recording = sd.rec(int(DURATION * SAMPLERATE), 
                   samplerate=SAMPLERATE, 
                   channels=CHANNELS, 
                   dtype='int16', 
                   device=DEVICE_ID)

sd.wait()  # aspetta fine registrazione
wav.write(filename, SAMPLERATE, recording)
print(f"Audio registrato in: {filename}")
