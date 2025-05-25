import sounddevice as sd
import soundfile as sf
import numpy as np

# === PARAMETRI COMPATIBILI CON IL MODELLO ===
SAMPLE_RATE = 22050    # Frequenza di campionamento (Hz)
DURATION = 4           # Durata della registrazione in secondi
CHANNELS = 1           # Mono
FILENAME = "recorded.wav"

print(f"🎙️ Registrazione in corso ({DURATION} secondi)...")

# Registra l'audio
recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
sd.wait()

print(f"💾 Salvataggio in '{FILENAME}'...")

# Salva l'audio in un file .wav
sf.write(FILENAME, recording, SAMPLE_RATE)

print("✅ Fatto. File pronto per la classificazione.")
