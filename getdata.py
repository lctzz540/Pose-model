import os
import cv2
import polars as pl
import mediapipe as mp
from parse import args


pose_estimator = mp.solutions.pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

video_dir = args.video_dir
csv_dir = args.csv_dir
done = 0
for filename in os.listdir(video_dir):
    if filename == ".DS_Store":
        continue
    video_name = os.path.splitext(filename)[0]

    cap = cv2.VideoCapture(os.path.join(video_dir, filename))
    frame_count = 0
    df = None
    while True:
        success, image = cap.read()
        if not success:
            break

        frame_count += 1

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(image)
        if results.pose_landmarks is None:
            continue

        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
            keypoints.append(landmark.z)
            keypoints.append(landmark.visibility)

        keypoints = pl.Series("keypoints", keypoints)
        new_column_name = f"keypoints_{frame_count}"
        keypoints = keypoints.rename(new_column_name)

        if df is None:
            df = keypoints.to_frame()
        else:
            df = df.hstack(keypoints.to_frame())
    try:
        df = df.transpose()
        df.write_csv(os.path.join(csv_dir, f"{video_name}.csv"))
        done += 1
    except:
        pass

print(f"Done {done}/{len(os.listdir(video_dir))}")
