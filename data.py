import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PoseDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        self.pose_df = pd.read_csv(csv_file)
        self.scaler = StandardScaler().fit(self.pose_df.iloc[:, 1:].values)

    def __len__(self):
        return len(self.pose_df)

    def __getitem__(self, idx):
        pose_coords = self.pose_df.iloc[idx, 1:].values
        pose_coords = pose_coords.astype("float32")
        pose_coords = torch.from_numpy(pose_coords)
        label = self.pose_df.iloc[idx, 0]
        return pose_coords, label
