import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Download training data from open datasets.
training_data = torchvision.datasets.CIFAR10(
    root="../dataset",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

# Download testing data from open datasets.
testing_data = torchvision.datasets.CIFAR10(
    root="../dataset",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

# Hyper parameters
batch_size = 10
EPOCHS = 10
learning_rate = 0.001

# Create data loaders.
train_dataloader = torch.utils.data.DataLoader(
    training_data,
    shuffle=True,
    batch_size=batch_size,
    num_workers=2
    )
test_dataloader = torch.utils.data.DataLoader(
    testing_data,
    shuffle=True,
    batch_size=batch_size,
    num_workers=2)

# Our model will recognize these kinds of objects
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Grab images from our training data
load_data_itr = iter(train_dataloader)
imgs, lables = load_data_itr.next()

print(classes[lables[0]])
plt.imshow(np.transpose(imgs[0], (1, 2, 0)))
plt.show()


# Get cpu or gpu device for training.
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define a convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.input_lay = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.headen1_lay = nn.Linear(in_features=120, out_features=84)
        self.output_lay = nn.Linear(in_features=84, out_features=10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.input_lay(x))
        x = F.relu(self.headen1_lay(x))
        x = self.output_lay(x)
        return x
net = Net()


# Define a loss function and optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# optimizer = optim.Adam(net.parameters(), lr=learning_rate)                    # things to try

print("Your network is ready for training!")


print("Training...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1} of {EPOCHS}", leave=True, ncols=80)):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()

# Save our trained model
PATH = '../models/cifar_net.pth'
torch.save(net.state_dict(), PATH)

# Pick random photos from training set
if load_data_itr == None:
    load_data_itr = iter(test_dataloader)
images, labels = load_data_itr.next()

# Load our model
net = Net()
net.load_state_dict(torch.load(PATH))

# Analyze images
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print(predicted)
# Show results
for i in range(batch_size):
    # Add new subplot
    plt.subplot(2, int(batch_size/2), i + 1)
    # Plot the image
    img = images[i]
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    # Add the image's label
    color = "green"
    label = classes[predicted[i]]
    if classes[labels[i]] != classes[predicted[i]]:
        color = "red"
        label = "(" + label + ")"
    plt.title(label, color=color)

plt.suptitle('Objects Found by Model', size=20)
plt.show()

# Measure accuracy for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
tot_accuracy = 0

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# Print accuracy statistics
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    tot_accuracy+=accuracy
print(f"total accuracy is {tot_accuracy/10}%")
