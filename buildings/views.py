from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Define paths to your model and class names
model_path = './buildings/trained_resnet18.pth'  # Make sure to update this path
class_names_path = './buildings/class_names.pth'  # Make sure to update this path

# Load the model and set it up for inference
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 6)  # Modify for 6 classes

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Load class names
    class_names = torch.load(class_names_path, map_location=device)

    return model, class_names, device

# Load the model, class names, and device globally
model, class_names, device = load_model()

# Define the image transformations used during training
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction view
@csrf_exempt
def predict(request):
    if request.method == 'POST' and request.FILES.get('building-image'):
        try:
            # Get the uploaded image
            image_file = request.FILES['building-image']

            # Open and transform the image
            image = Image.open(image_file)
            image = transform(image).unsqueeze(0)  # Add batch dimension

            # Move the image to the correct device (GPU or CPU)
            image = image.to(device)

            # Get the model's prediction
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            # Map the predicted class index to a human-readable label
            predicted_class_name = class_names[predicted.item()]

            # # Example: You can also return the class name as part of the building data
            # building_data = {
            #     "name": predicted_class_name,
            # }

            # return render(request, 'building.html', {'building_data': building_data})
            return JsonResponse({'name': predicted_class_name}, status=200)

        except Exception as e:
            return JsonResponse({'error': f'Failed to process image: {str(e)}'}, status=500)

    return render(request, 'building.html')


def index_view(request):
    return render(request, 'index.html')  # Ensure 'index.html' exists in your templates


def building_view(request):
    return render(request, 'building.html')  # Ensure 'index.html' exists in your templates