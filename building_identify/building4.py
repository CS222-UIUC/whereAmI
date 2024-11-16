
from torchvision import models, transforms
import torch

from torchvision.datasets import ImageFolder


# Define data transformations with data augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Assuming `train_dataset` is the ImageFolder dataset
train_dataset = ImageFolder(root='data/images/train', transform=transform)
class_names = train_dataset.classes

# Save class names to a file (could be any path you prefer)
torch.save(class_names, '/Users/evelynzhou/Documents/Code/cs222/whereAmI/building_identify/class_names.pth')