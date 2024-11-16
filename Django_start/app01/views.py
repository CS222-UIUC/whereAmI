# views.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
import os
from django.conf import settings

# Get the path relative to the Django project base directory
model_path = os.path.join(settings.BASE_DIR, 'app01', 'static', 'models', 'trained_resnet18.pth')
class_names_path = os.path.join(settings.BASE_DIR, 'app01', 'static', 'models', 'class_names.pth')

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 6)  # Modify for 6 classes
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Load class names
class_names = torch.load(class_names_path, weights_only=True)


# Define the image transformations used during training
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Prediction view
def predict(request):
    if request.method == 'POST' and request.FILES.get('building-image'):
        image_file = request.FILES['building-image']

        # Open the image
        image = Image.open(image_file)
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Move the image to the right device (GPU or CPU)
        image = image.to(device)

        # Get the model's prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Map the predicted class index to a human-readable label
        predicted_class_name = class_names[predicted.item()]

        # Example: You can also return the class name as part of the building data
        building_data = {
            "name": predicted_class_name,
            "bus_stops": "N/A",  # Replace with actual data or logic
            "functions": "N/A",  # Replace with actual data or logic
        }

        return render(request, 'building.html', {'building_data': building_data})


    return render(request, 'building.html')

