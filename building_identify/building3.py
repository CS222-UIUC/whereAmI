import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Unfreeze the last two residual blocks and fully connected layer for more fine-tuning
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Modify the final fully connected layer to fit 6 output classes
num_classes = 6
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer with a lower learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

# Use ReduceLROnPlateau to reduce learning rate when validation loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Training with early stopping
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

best_loss = float('inf')
early_stopping_patience = 5  # Increased patience for early stopping
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

    # Step the learning rate scheduler
    scheduler.step(validation_loss)

    # Check for early stopping
    if validation_loss < best_loss:
        best_loss = validation_loss
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stopping_patience:
            print("Early stopping")
            break
# Save the model after training
torch.save(model.state_dict(), 'trained_resnet50.pth')

