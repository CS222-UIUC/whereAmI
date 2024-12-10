import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
# Prepare dataset
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# Length of the datasets
train_data_size = len(train_data)
test_data_size = len(test_data)
# If train_data_size=10, then the length of the training dataset is: 10
print("Length of training dataset: {}".format(train_data_size))
print("Length of test dataset: {}".format(test_data_size))


# Use DataLoader to load the dataset
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Create network model
Process = Process()

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
# learning_rate = 0.01
# 1e-2 = 1 x (10)^(-2) = 1 / 100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(Process.parameters(), lr=learning_rate)

# Set some parameters for training the network
# Record the training steps
total_train_step = 0
# Record the test steps
total_test_step = 0
# Number of epochs
epoch = 10

# Add tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("------- Start of epoch {} -------".format(i+1))

    # Start training steps
    Process.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = Process(imgs)
        loss = loss_fn(outputs, targets)

        # Optimizer updates the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("Training step: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # Start test steps
    Process.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = Process(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("Total Loss on test dataset: {}".format(total_test_loss))
    print("Total Accuracy on test dataset: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(Process, "Process_{}.pth".format(i))
    print("Model saved")

writer.close()
