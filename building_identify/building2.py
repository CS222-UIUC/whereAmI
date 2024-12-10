import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Define data transformations with data augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(root='data/images/train', transform=transform)
test_dataset = ImageFolder(root='data/images/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Unfreeze the last residual block and fully connected layer
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Modify the final fully connected layer to fit 6 output classes
num_classes = 6
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Training with early stopping
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

best_loss = float('inf')
patience = 3  # Number of epochs to wait for improvement
no_improve_epochs = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation step
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Check for early stopping
    if validation_loss < best_loss:
        best_loss = validation_loss
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping")
            break

    # Step the learning rate scheduler
    scheduler.step()

# Initialize dictionaries to store per-class results
class_correct = {i: 0 for i in range(num_classes)}
class_total = {i: 0 for i in range(num_classes)}
individual_results = []  # List to store individual image results

# Final evaluation on test set with per-class and individual accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Update overall accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i].item()
            pred = predicted[i].item()
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1

            # Store individual image results
            individual_results.append({
                'true_label': label,
                'predicted_label': pred,
                'correct': label == pred
            })

# Overall accuracy
print(f'\nFinal Accuracy on test set: {100 * correct / total:.2f}%')

# Per-class accuracy
print("\nPer-class accuracy:")
for cls in range(num_classes):
    if class_total[cls] > 0:
        accuracy = 100 * class_correct[cls] / class_total[cls]
        print(f'Accuracy of class {cls}: {accuracy:.2f}%')
    else:
        print(f'No samples found for class {cls}')

# Individual image results
print("\nIndividual image predictions:")
for i, result in enumerate(individual_results):
    print(f"Image {i + 1}: True Label = {result['true_label']}, Predicted Label = {result['predicted_label']}, Correct = {result['correct']}")
