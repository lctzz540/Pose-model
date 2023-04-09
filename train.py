import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import PoseDataset
from network import LSTMPoseClassifier
from parse import args
from graph import exportgraph


device = torch.device(args.device)

input_size = 131
hidden_size = 128
output_size = 1
batch_size = 2
learning_rate = 0.001
num_epochs = args.epochs
patience = 3


train_dataset = PoseDataset(args.train_data_folder)
valid_dataset = PoseDataset(args.valid_data_folder)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = LSTMPoseClassifier(input_size, hidden_size, output_size, num_classes=1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.BCEWithLogitsLoss()

best_loss = float("inf")
counter = 0


# Create empty lists to store loss and accuracy values
train_loss = []
train_acc = []
valid_losses = []
valid_acc = []

# Train loop


for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        r_, predicted = torch.max(outputs.data, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * running_correct / running_total
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    print(
        f"Epoch {epoch+1} train loss: {epoch_loss:.4f}, train accuracy: {epoch_acc:.2f}%"
    )

    # Evaluate on validation set
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float().view(-1, 1))
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    valid_loss /= len(valid_loader)
    valid_losses.append(valid_loss)
    valid_acc = 100.0 * valid_correct / valid_total
    print(
        f"Validation loss: {valid_loss:.4f}, Validation accuracy: {valid_acc:.2f}%")

    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping after {epoch+1} epochs")
        break

exportgraph(train_loss, valid_losses, train_acc, valid_acc)
