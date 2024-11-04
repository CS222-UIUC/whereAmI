from fastai.vision.all import *

# Define the DataBlock(DataLoaders that contains a training set and a validation set)
dls = DataBlock(
    # ImageBlock is input and 1, 2, 3, 4 etc are output, so they are categories.
    # cls = PILImageBW is a class for grayscale images, which is MNIST images
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
    # Find all the inputs to our model
    get_items=get_image_files, 
    # Split into 20 percent validation set and 80 percent training set
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    # The labels (y values) is the name of the parent of each file
    get_y=parent_label, 
    # Resize all images to 28x28 pixels as standard MNIST
    item_tfms=Resize(28), 
    # Normalize the images
    batch_tfms=Normalize() 
).dataloaders(untar_data(URLs.MNIST))

# Set n_in=1 because MNIST images have 1 channel(Gray)
learn = vision_learner(dls, resnet18, metrics=accuracy, n_in=1)

# The method automatically uses best practices for fine tuning a pre-trained model
# Train for five epochs
learn.fine_tune(5)


# Save the trained model, can load it in future use
learn.save('mnist_training_fastai')

# Result of the model: 
# epoch     train_loss  valid_loss  accuracy  time    
# 0         0.138810    0.092094    0.971357  02:53                                                                                          
# 1         0.087267    0.049692    0.985214  02:51                                                                                          
# 2         0.046611    0.035852    0.988357  02:51                                                                                          
# 3         0.024782    0.034051    0.990857  02:50                                                                                          
# 4         0.010046    0.035038    0.990500  02:50  

# Achieves an accuracy of about 0.99
