import sys
import librosa
import numpy as np
from tensorflow.keras.models import load_model

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
