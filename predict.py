import torch
from network import LSTMPoseClassifier
from data import PoseDataset
from parse import args
from torch.utils.data import DataLoader


device = torch.device(args.device)

model = LSTMPoseClassifier(131, 128, 1, num_classes=1)
model.load_state_dict(torch.load("best_model.pt"))
model.to(device)
model.eval()

dataset = PoseDataset(args.test_data_folder)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        print(f"Predicted: {predicted}, Ground Truth: {labels}")
