import os
import json
import pandas as pd
import numpy as np

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
            # Extract basic dicts
            posture = p.get("posture", p.get("scores", {}))
            gaze = p.get("gaze", {})
            headDir = gaze.get("headDirection", {})
            emo = p.get("emotion", {})
            hnds = p.get("hands", {})

            # Features
            overall = posture.get("overall", 0)
            shoulderTilt = posture.get("shoulderTilt", 0)
            headForward = posture.get("headForward", 0)
            bodyLean = posture.get("bodyLean", 0)

            yaw = headDir.get("yaw", 0)
            pitch = headDir.get("pitch", 0)
            roll = headDir.get("roll", 0)

            frown = emo.get("frown", 0)
            smile = emo.get("smile", 0)
            eyeOpenness = emo.get("eyeOpenness", 1.0)

            handsVisible = hnds.get("handsVisible", 0)
            isTouchingFace = 1 if hnds.get("isTouchingFace", False) else 0

            # Label logic based on JS rules
            # Pattern A: Confusion
            is_confusion = isTouchingFace == 1 or frown > 0.2
            # Pattern B: Distraction
            is_head_turned = abs(yaw) > 25 or abs(pitch) > 25
            is_distraction = eyeOpenness < 0.4 or is_head_turned
            # Pattern C: Bad Posture
            is_bad_posture = overall < 60
            # Pattern D: Flow
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

            # Feature vector
            feat = [
                overall, shoulderTilt, headForward, bodyLean,
                yaw, pitch, roll,
                frown, smile, eyeOpenness,
                handsVisible, isTouchingFace
            ]
            
            features.append(feat)
            labels.append(label)

    columns = [
        "overall", "shoulderTilt", "headForward", "bodyLean",
        "yaw", "pitch", "roll",
        "frown", "smile", "eyeOpenness",
        "handsVisible", "isTouchingFace"
    ]

    df = pd.DataFrame(features, columns=columns)
    df["label"] = labels
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
