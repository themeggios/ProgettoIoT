import librosa
import numpy as np
from tensorflow.keras.models import load_model

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

X = estrai_mfcc("test.wav")

print("Forma input:", X.shape)
pred = model.predict(X)
classe = np.argmax(pred)
print("Classe predetta:", classe)
