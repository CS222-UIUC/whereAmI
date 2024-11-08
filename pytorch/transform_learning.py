import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time

# Directories for training, validation, and test data
data_dir = '/Users/jay/Desktop/CS/CS222/whereAmI/pytorch/data'
train_dir = os.path.join(data_dir, 'training')
valid_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'testing')

# Data Augmentation
data_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        # Randomly changes the brightness, contrast, saturation and hue of the image
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Load data from folder
data = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

# Create data loaders
batch_size = 32
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'valid': DataLoader(data['valid'], batch_size=batch_size, shuffle=False),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=False)
}

# Load pretrained ResNet50 Model
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 5), 
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)

# Define Optimizer and Loss Function
loss_func = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# Set the number of epochs
epochs = 10
# Loop through the epochs
for epoch in range(epochs):
    start_time = time.time()
    print(f"Epoch {epoch+1}/{epochs}")

    # Set model to training mode
    model.train()

    # Initialize training loss and accuracy for the epoch
    train_loss = 0.0
    train_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    # Training loop
    for i, (inputs, labels) in enumerate(dataloaders['train']):
        # Zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch + 1} completed in {time.time() - start_time:.2f}s')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct // total} %')
    

# Calculate the final test accuracy using the test images
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final Test Accuracy: {100 * correct // total:.2f}%')
# Approximate accuracy: 0.9