import os
import torch
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class PoseDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.activity_files = []
        self.label_map = {}
        label = 0
        data = []
        for activity_folder in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, activity_folder)):
                for filename in os.listdir(os.path.join(data_dir, activity_folder)):
                    if filename.endswith(".csv"):
                        activity_name = activity_folder
                        self.activity_files.append(
                            os.path.join(data_dir, activity_folder, filename)
                        )
                        if activity_name not in self.label_map:
                            self.label_map[activity_name] = label
                            label += 1
                        activity_df = pd.read_csv(
                            os.path.join(data_dir, activity_folder, filename)
                        )
                        activity_data = activity_df.iloc[:, 1:].values.astype(
                            "float32")
                        data.append(activity_data)
        data = np.concatenate(data, axis=0)
        self.scaler = StandardScaler()
        self.scaler.fit(data)

    def __len__(self):
        return len(self.activity_files)

    def __getitem__(self, idx):
        activity_file = self.activity_files[idx]
        activity_name = os.path.basename(os.path.dirname(activity_file))
        activity_df = pd.read_csv(activity_file)
        row_idx = random.randint(0, len(activity_df) - 1)
        pose_coords = activity_df.iloc[row_idx, 1:].values
        pose_coords = pose_coords.astype("float32")
        pose_coords = self.scaler.transform(pose_coords.reshape(1, -1))
        pose_coords = torch.from_numpy(pose_coords)
        label = self.label_map[activity_name]
        return pose_coords, label
