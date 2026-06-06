import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Transforms
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------
# Load full dataset
# ------------------------
full_dataset = datasets.ImageFolder("dataset", transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)

targets = np.array(full_dataset.targets)
indices = np.arange(len(targets))

# ------------------------
# Stratified Split: Train / Val / Test
# ------------------------
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.3,
    stratify=targets,
    random_state=42
)

temp_targets = targets[temp_idx]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=temp_targets,
    random_state=42
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

# ------------------------
# DataLoaders
# ------------------------
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# ------------------------
# Model
# ------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ------------------------
# Loss & Optimizer
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------------------
# Metrics storage
# ------------------------
train_losses = []
val_accuracies = []

# ------------------------
# Training loop
# ------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ------------------------
    # Validation
    # ------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {avg_train_loss:.4f} "
          f"Val Acc: {val_acc:.4f}")

# ------------------------
# Test Evaluation
# ------------------------
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ------------------------
# Classification Report
# ------------------------
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ------------------------
# Confusion Matrix
# ------------------------
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# ------------------------
# Plot Loss & Accuracy
# ------------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(val_accuracies, label="Val Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.show()

# ------------------------
# Save Model
# ------------------------
output_path = Path("models") / "resnet18_car.pth"
output_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), output_path)

print("\nModel saved at:", output_path)