# views.py
from django.http import JsonResponse
from .ml_model import classify_image
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render


@csrf_exempt  
def upload_image(request):
    if request.method == 'POST':
        if 'building-image' not in request.FILES:
            return JsonResponse({'error': 'No image file provided'}, status=400)

        image_file = request.FILES['building-image']

        try:
            # Classify the uploaded image
            prediction = classify_image(image_file)
            return JsonResponse(prediction)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def index_view(request):
    return render(request, 'index.html')

def building_view(request):
    return render(request, 'building.html')
