import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import PoseDataset
from network import LSTMPredictor

# Define command line arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu",
                    help="device to train on")
parser.add_argument(
    "--epochs", type=int, default=10, help="number of epochs to train for"
)
args = parser.parse_args()

# Set device
device = torch.device(args.device)

# Hyperparameters
input_size = 131
hidden_size = 128
output_size = 6
batch_size = 32
learning_rate = 0.001
num_epochs = args.epochs
patience = 3

# Load data
train_dataset = PoseDataset("./dataset/TreePoseRight.csv")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = LSTMPredictor(input_size, hidden_size, output_size)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Train model
best_loss = float("inf")
counter = 0

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} loss: {epoch_loss}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping after {epoch+1} epochs")
        break

print("Finished Training")
