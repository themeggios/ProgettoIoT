import os
import time
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense
)

# — CONFIGURAZIONE —
DATASET_PATH = "UrbanSound8K"
META_CSV     = os.path.join(DATASET_PATH, "metadata/UrbanSound8K.csv")
AUDIO_DIR    = os.path.join(DATASET_PATH, "audio")
SR           = 22050        # sample rate
DUR          = 4.0          # durata in secondi per clip
N_MFCC       = 40           # numero di coefficienti MFCC
MAX_LEN      = 174          # numero di frame (colonne) per MFCC

# — FUNZIONE DI ESTRAZIONE MFCC 2D —
def extract_mfcc_2d(file_path, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    y, _ = librosa.load(file_path, sr=sr, duration=DUR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # pad o trim in colonne
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# — Caricamento dati e label —
meta = pd.read_csv(META_CSV)
X, labels = [], []

for _, row in meta.iterrows():
    fold = f"fold{row['fold']}"
    file_path = os.path.join(AUDIO_DIR, fold, row['slice_file_name'])
    try:
        mfcc2d = extract_mfcc_2d(file_path)
        X.append(mfcc2d)
        labels.append(row['classID'])
    except Exception:
        continue

X = np.array(X)[..., np.newaxis]  # shape: (n_samples, n_mfcc, max_len, 1)
labels = np.array(labels)

# — Encoding delle etichette e one-hot —
le = LabelEncoder()
y_int = le.fit_transform(labels)
y = to_categorical(y_int)

# — Split train/test —
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_int, random_state=42
)

# — Costruzione del modello CNN —
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(y.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# — Training con misurazione del tempo —
start = time.time()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

end = time.time()
print(f"\n✅ Training completato in {(end - start)/60:.2f} minuti")

# — Valutazione su test set —
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.3f}, Test Loss: {test_loss:.3f}")

# — Salvataggio modello e label encoder —
model.save("cnn_audio_classifier.h5")
import joblib
joblib.dump(le, "label_encoder.pkl")

print("✅ Modello e label encoder salvati.")
