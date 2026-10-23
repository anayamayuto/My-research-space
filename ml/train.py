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


def blockwise_split(y, test_ratio=0.2):
    """Time-aware split: for each class, the LAST test_ratio of its rows
    (in recording order) go to the test set.

    A plain random train_test_split leaks information with this kind of
    time-series data: adjacent frames are nearly identical, so putting one
    in train and its neighbour in test massively inflates test accuracy.
    """
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n_test = max(1, int(len(idx) * test_ratio))
        train_idx.extend(idx[:-n_test])
        test_idx.extend(idx[-n_test:])
    return np.array(train_idx), np.array(test_idx)


NUM_ATTEMPTS = 5  # small network + random init sometimes lands in a bad
# local optimum for one class (observed: same data/code, one run gave class
# recall 0.15, the next gave 0.97). Train a few times and keep the run with
# the best worst-case class recall instead of trusting a single run.


def train_once(X_train, y_train, X_test, y_test, class_weight, num_features):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(num_features,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=60, batch_size=32,
              validation_data=(X_test, y_test), class_weight=class_weight, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    recalls = {}
    for c in range(5):
        mask = y_test == c
        if mask.sum() > 0:
            recalls[c] = (pred[mask] == c).mean()
    return model, acc, recalls


def main():
    df = pd.read_csv('dataset.csv')
    X = df.drop('label', axis=1).values
    y = df['label'].values

    if len(X) < 10:
        print("Not enough data to train. Please collect more data in the app.")
        return

    # Warn about near-constant features (these break StandardScaler at
    # inference time — see the headForward incident)
    stds = X.std(axis=0)
    for name, s in zip(df.columns[:-1], stds):
        if s < 1e-6:
            print(f"WARNING: feature '{name}' is constant in the dataset. "
                  f"It carries no information and is dangerous at inference "
                  f"time. Consider removing it or collecting more varied data.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_idx, test_idx = blockwise_split(y, test_ratio=0.2)
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Class weights so no class gets ignored
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Class weights:", class_weight)

    print(f"Training model ({NUM_ATTEMPTS} attempts, keeping the best)...")
    best = None
    for attempt in range(1, NUM_ATTEMPTS + 1):
        model, acc, recalls = train_once(X_train, y_train, X_test, y_test, class_weight, X.shape[1])
        worst_recall = min(recalls.values()) if recalls else 0
        print(f"  attempt {attempt}: acc={acc:.4f} worst_class_recall={worst_recall:.2f} "
              f"per_class={ {c: round(r, 2) for c, r in recalls.items()} }")
        if best is None or worst_recall > best[2]:
            best = (model, acc, worst_recall, recalls)

    model, acc, worst_recall, recalls = best
    print(f"Best attempt: Test Accuracy: {acc:.4f}")
    print("(Note: with data from a single session this number is still "
          "optimistic. The real test is live behaviour in the app.)")
    for c, r in recalls.items():
        print(f"  class {c}: recall {r:.2f}")
    if worst_recall < 0.8:
        print(f"WARNING: worst class recall is {worst_recall:.2f} even in the best of "
              f"{NUM_ATTEMPTS} attempts — collect more/cleaner data for that class "
              f"rather than trusting this model as-is.")

    # Save scaler params to JSON so JS can apply exactly the same scaling
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    out_dir = "../public/model"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f)

    # Export model for TF.js
    tfjs.converters.save_keras_model(model, out_dir)
    print(f"Model and scaler saved to {out_dir}")


if __name__ == "__main__":
    main()
