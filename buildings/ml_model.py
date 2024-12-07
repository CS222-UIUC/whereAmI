# ml_model.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Load the pre-trained ResNet-18 model and set it to evaluation mode
num_classes = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load the model
def load_model():
    model = models.resnet18(pretrained=False)  
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('trained_resnet18.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Function to classify an image
def classify_image(image_file):
    # Load the model
    model = load_model()

    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Open the image, apply the transformations, and make a prediction
    image = Image.open(image_file).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Convert prediction to class label
    class_idx = predicted.item()
    class_labels = ['Engineering Hall', 'Everitt Lab', 'Grainger Library', 'Material Science Building', 'Mechanical Engineering Building', 'Siebel Center for Computer Science']  
    return {'building_name': class_labels[class_idx]}
