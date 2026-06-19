import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflowjs as tfjs
import json
import os

def main():
    # Load dataset
    df = pd.read_csv('dataset.csv')
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Check if we have enough data
    if len(X) < 10:
        print("Not enough data to train. Please collect more data in the app.")
        return

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Simple model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

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
