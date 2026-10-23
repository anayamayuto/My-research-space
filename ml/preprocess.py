import os
import json
import math
import pandas as pd
import numpy as np

# Feature order MUST match buildFeatureVector() in lecture-client-ml.js
# and the CSV header in server.js /api/annotation-data.
#
# Changes from v1:
# - headForward removed: it was constant 0 in the collected data, which made
#   StandardScaler store mean=0/scale=1; at runtime any non-zero value became
#   a huge z-score that dominated the model output.
# - smile removed: near-constant (~0), its tiny std amplified noise.
# - roll is circular (wraps at +/-180 deg), so it is encoded as sin/cos.
COLUMNS = [
    "overall", "shoulderTilt", "bodyLean",
    "yaw", "pitch", "rollSin", "rollCos",
    "frown", "eyeOpenness",
    "handsVisible", "isTouchingFace",
]


def process_data(data_dir):
    features = []
    labels = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {filename}")
            continue

        points = data.get("dataPoints", [])
        for p in points:
            posture = p.get("posture", p.get("scores", {}))
            gaze = p.get("gaze", {})
            headDir = gaze.get("headDirection", {})
            emo = p.get("emotion", {})
            hnds = p.get("hands", {})

            overall = posture.get("overall", 0)
            shoulderTilt = posture.get("shoulderTilt", 0)
            bodyLean = posture.get("bodyLean", 0)

            yaw = headDir.get("yaw", 0)
            pitch = headDir.get("pitch", 0)
            roll = headDir.get("roll", 0)
            roll_rad = math.radians(roll)

            frown = emo.get("frown", 0)
            eyeOpenness = emo.get("eyeOpenness", 1.0)

            handsVisible = hnds.get("handsVisible", 0)
            isTouchingFace = 1 if hnds.get("isTouchingFace", False) else 0

            # Label logic based on JS rules (used only for auto-labelling
            # pose_data sessions; manually annotated data from the app
            # already carries its own labels)
            is_confusion = isTouchingFace == 1 or frown > 0.2
            is_head_turned = abs(yaw) > 25 or abs(pitch) > 25
            is_distraction = eyeOpenness < 0.4 or is_head_turned
            is_bad_posture = overall < 60
            is_flow = overall >= 70 and not is_head_turned

            # Priority: Confusion (1) > Distraction (2) > Bad Posture (3) > Flow (4) > Normal (0)
            label = 0
            if is_confusion:
                label = 1
            elif is_distraction:
                label = 2
            elif is_bad_posture:
                label = 3
            elif is_flow:
                label = 4

            feat = [
                overall, shoulderTilt, bodyLean,
                yaw, pitch,
                math.sin(roll_rad), math.cos(roll_rad),
                frown, eyeOpenness,
                handsVisible, isTouchingFace,
            ]

            features.append(feat)
            labels.append(label)

    df = pd.DataFrame(features, columns=COLUMNS)
    df["label"] = labels
    return df


def convert_old_dataset(old_csv_path):
    """Convert a v1-format dataset.csv (12 features incl. headForward, roll,
    smile) into the new 11-feature format, preserving labels."""
    old = pd.read_csv(old_csv_path)
    roll_rad = np.radians(old["roll"])
    df = pd.DataFrame({
        "overall": old["overall"],
        "shoulderTilt": old["shoulderTilt"],
        "bodyLean": old["bodyLean"],
        "yaw": old["yaw"],
        "pitch": old["pitch"],
        "rollSin": np.sin(roll_rad),
        "rollCos": np.cos(roll_rad),
        "frown": old["frown"],
        "eyeOpenness": old["eyeOpenness"],
        "handsVisible": old["handsVisible"],
        "isTouchingFace": old["isTouchingFace"],
        "label": old["label"],
    })
    return df


if __name__ == "__main__":
    pose_data_dir = "../pose_data"
    if not os.path.exists(pose_data_dir):
        print(f"Error: {pose_data_dir} not found.")
        exit(1)

    df = process_data(pose_data_dir)
    print(f"Processed {len(df)} samples.")
    print("Label distribution:")
    print(df["label"].value_counts())

    out_path = "dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}")
