import os
from PIL import Image
import pillow_heif

# Set folder path
folder_path = 'data/images/train'
#folder_path = 'data/images/test'

# Traverse all subfolders
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        # Check if the file ends with .HEIC
        if filename.lower().endswith('.heic'):
            heic_file = os.path.join(root, filename)
            jpg_file = os.path.join(root, filename.replace('.HEIC', '.jpg'))

            # Open the HEIC file and convert to JPG
            heif_file = pillow_heif.open_heif(heic_file)
            image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)

            # Save as JPG format
            image.save(jpg_file, "JPEG")

            # Delete the original HEIC file
            os.remove(heic_file)

print("Conversion from HEIC to JPG format completed!")
