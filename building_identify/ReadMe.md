## Accuracy using building1.py:
> Epoch 1/10, Loss: 1.9581187963485718  
> Epoch 2/10, Loss: 1.5919998586177826  
> Epoch 3/10, Loss: 1.3849587440490723  
> Epoch 4/10, Loss: 1.280463993549347  
> Epoch 5/10, Loss: 1.1038118302822113  
> Epoch 6/10, Loss: 0.9529491811990738  
> Epoch 7/10, Loss: 0.9063521325588226  
> Epoch 8/10, Loss: 0.8262439072132111  
> Epoch 9/10, Loss: 0.8893081992864609  
> Epoch 10/10, Loss: 0.7293008416891098  

**Accuracy on test set:** 71.43%  

## Accuracy using building2.py:
> Epoch 1/20, Training Loss: 1.7745, Validation Loss: 1.6376, Accuracy: 35.71%  
> Epoch 2/20, Training Loss: 1.5528, Validation Loss: 1.5244, Accuracy: 50.00%  
> Epoch 3/20, Training Loss: 1.3732, Validation Loss: 1.4154, Accuracy: 64.29%  
> Epoch 4/20, Training Loss: 1.3039, Validation Loss: 1.3463, Accuracy: 57.14%  
> Epoch 5/20, Training Loss: 1.1774, Validation Loss: 1.4109, Accuracy: 50.00%  
> Epoch 6/20, Training Loss: 1.0714, Validation Loss: 1.4011, Accuracy: 57.14%  
> Epoch 7/20, Training Loss: 1.1200, Validation Loss: 1.3278, Accuracy: 57.14%  
> Epoch 8/20, Training Loss: 1.1653, Validation Loss: 1.4417, Accuracy: 57.14%  
> Epoch 9/20, Training Loss: 1.1766, Validation Loss: 1.4377, Accuracy: 57.14%  
> Epoch 10/20, Training Loss: 1.1614, Validation Loss: 1.3133, Accuracy: 57.14%  
> Epoch 11/20, Training Loss: 1.1203, Validation Loss: 1.3532, Accuracy: 57.14%  
> Epoch 12/20, Training Loss: 1.0894, Validation Loss: 1.3476, Accuracy: 57.14%  
> Epoch 13/20, Training Loss: 1.1082, Validation Loss: 1.3983, Accuracy: 57.14%  
> Early stopping  

**Final Accuracy on test set:** 57.14%  

**Per-class accuracy:**  
- Accuracy of class 0: 100.00%  
- Accuracy of class 1: 50.00%  
- Accuracy of class 2: 100.00%  
- Accuracy of class 3: 0.00%  
- Accuracy of class 4: 0.00%  
- Accuracy of class 5: 0.00%  

**Individual image predictions:**  
- Image 1: True Label = 0, Predicted Label = 0, Correct = True  
- Image 2: True Label = 0, Predicted Label = 0, Correct = True  
- Image 3: True Label = 1, Predicted Label = 1, Correct = True  
- Image 4: True Label = 1, Predicted Label = 2, Correct = False  
- Image 5: True Label = 2, Predicted Label = 2, Correct = True  
- Image 6: True Label = 2, Predicted Label = 2, Correct = True  
- Image 7: True Label = 2, Predicted Label = 2, Correct = True  
- Image 8: True Label = 2, Predicted Label = 2, Correct = True  
- Image 9: True Label = 2, Predicted Label = 2, Correct = True  
- Image 10: True Label = 3, Predicted Label = 1, Correct = False  
- Image 11: True Label = 3, Predicted Label = 2, Correct = False  
- Image 12: True Label = 4, Predicted Label = 1, Correct = False  
- Image 13: True Label = 5, Predicted Label = 2, Correct = False  
- Image 14: True Label = 5, Predicted Label = 1, Correct = False  


## Accuracy using building3.py:
> Epoch 1/20, Training Loss: 1.9479, Validation Loss: 1.6967, Accuracy: 35.71%  
> Epoch 2/20, Training Loss: 1.6980, Validation Loss: 1.4989, Accuracy: 50.00%  
> Epoch 3/20, Training Loss: 1.5614, Validation Loss: 1.5637, Accuracy: 35.71%  
> Epoch 4/20, Training Loss: 1.4032, Validation Loss: 1.4840, Accuracy: 42.86%  
> Epoch 5/20, Training Loss: 1.3765, Validation Loss: 1.4389, Accuracy: 57.14%  
> Epoch 6/20, Training Loss: 1.1937, Validation Loss: 1.1570, Accuracy: 50.00%  
> Epoch 7/20, Training Loss: 1.0769, Validation Loss: 1.0352, Accuracy: 85.71%  
> Epoch 8/20, Training Loss: 0.9663, Validation Loss: 1.0555, Accuracy: 78.57%  
> Epoch 9/20, Training Loss: 0.8932, Validation Loss: 0.9819, Accuracy: 71.43%  
> Epoch 10/20, Training Loss: 0.7572, Validation Loss: 0.9526, Accuracy: 71.43%  
> Epoch 11/20, Training Loss: 0.6976, Validation Loss: 0.8192, Accuracy: 78.57%  
> Epoch 12/20, Training Loss: 0.6484, Validation Loss: 0.7733, Accuracy: 78.57%  
> Epoch 13/20, Training Loss: 0.7160, Validation Loss: 0.7736, Accuracy: 78.57%  
> Epoch 14/20, Training Loss: 0.5654, Validation Loss: 0.7160, Accuracy: 78.57%  
> Epoch 15/20, Training Loss: 0.5184, Validation Loss: 0.7065, Accuracy: 78.57%  
> Epoch 16/20, Training Loss: 0.5440, Validation Loss: 0.5747, Accuracy: 85.71%  
> Epoch 17/20, Training Loss: 0.4460, Validation Loss: 0.5384, Accuracy: 92.86%  
> Epoch 18/20, Training Loss: 0.5161, Validation Loss: 0.5990, Accuracy: 92.86%  
> Epoch 19/20, Training Loss: 0.5356, Validation Loss: 0.6148, Accuracy: 85.71%  
> Epoch 20/20, Training Loss: 0.3196, Validation Loss: 0.5672, Accuracy: 85.71%  


## helper_rename.py
a helper function to change all the image suffix name to `.jpg`