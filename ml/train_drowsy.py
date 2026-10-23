"""Train the drowsiness model from self-reported labelled window features.

Data source: drowsiness_data/dataset.csv, collected in the app via the
"眠気データ記録" panel (one row every 5 seconds while recording).

Labels: 0 = 覚醒, 1 = 少し眠い, 2 = 眠い

Usage:
    cd ml
    python train_drowsy.py
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflowjs as tfjs
import json

NUM_CLASSES = 3

# Must match DROWSY.FEATURE_NAMES in lecture-client-ml.js and the CSV header
# written by server.js /api/drowsiness-data
FEATURES = [
    "perclos", "blinkRate", "meanBlinkDur", "longBlinks", "yawns",
    "nods", "pitchStd", "pitchDrift", "postureStd", "elapsedMin",
]


def blockwise_split(y, test_ratio=0.2):
    """Per-class contiguous split (last test_ratio of each class -> test).

    Rows 5 seconds apart share most of their 60s window, so a random split
    would leak train information into the test set and inflate accuracy.
    """
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n_test = max(1, int(len(idx) * test_ratio))
        train_idx.extend(idx[:-n_test])
        test_idx.extend(idx[-n_test:])
    return np.array(train_idx), np.array(test_idx)


def main():
    csv_path = '../drowsiness_data/dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} が見つかりません。"
              f"アプリの「眠気データ記録」でデータを集めてください。")
        return

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + ['label'] if c not in df.columns]
    if missing:
        print(f"Error: CSVに列が足りません: {missing}")
        return

    X = df[FEATURES].values
    y = df['label'].values.astype(int)

    print(f"Samples: {len(X)}")
    print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))

    if len(X) < 100:
        print("データが少なすぎます (最低100行、できれば各クラス300行以上)。")
        return
    if len(np.unique(y)) < 2:
        print("ラベルが1種類しかありません。覚醒/眠い 両方のセッションを記録してください。")
        return

    for name, s in zip(FEATURES, X.std(axis=0)):
        if s < 1e-6:
            print(f"WARNING: '{name}' が定数です。収集条件を見直してください。")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_idx, test_idx = blockwise_split(y, test_ratio=0.2)
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Class weights:", class_weight)

    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(len(FEATURES),)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training drowsiness model...")
    model.fit(X_train, y_train, epochs=80, batch_size=32,
              validation_data=(X_test, y_test), class_weight=class_weight)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    pred = model.predict(X_test).argmax(axis=1)
    for c in range(NUM_CLASSES):
        mask = y_test == c
        if mask.sum() > 0:
            print(f"  class {c}: recall {(pred[mask] == c).mean():.2f} ({mask.sum()} samples)")

    out_dir = "../public/model_drowsy"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "scaler_params.json"), "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

    tfjs.converters.save_keras_model(model, out_dir)
    print(f"Drowsiness model saved to {out_dir}")
    print("ブラウザをリロードすると自動で読み込まれます。")


if __name__ == "__main__":
    main()
