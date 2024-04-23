import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import cv2
from torchvision.utils import save_image
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomCrop, ColorJitter, GaussianBlur, RandomVerticalFlip

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([transforms.Resize((227,227)),transforms.ToTensor()])

# CIFAR dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True)

len(train_loader)

for image, _ in train_loader:
    image = image.to(device)  # Move data to GPU if available
    print(image.shape, torch.min(image), torch.max(image))
    break

min_sigma = 0
max_sigma = 50/255

def add_noise(image, sigma):
    noise = torch.randn_like(image) * sigma
    noisy_image = torch.clamp(image + noise, 0, 1)
    return noisy_image

def save_noisy_image(img, name):
    img = img.view(img.size(0), 3, 227, 227)
    save_image(img, name)

def customFunction():
    sigma_values = []
    i=1
    j=1
    # Loop through the dataset
    for inputs, _ in train_loader:

        # Generate random sigma within the specified range
        sigma = np.random.uniform(min_sigma, max_sigma)
        sigma_values.append(sigma)

        # Add noise to the input image
        noisy_inputs = add_noise(inputs, sigma)

        # Convert tensor to numpy array and save the noisy image
        noisy_image = (noisy_inputs.squeeze().numpy() * 255).astype(np.uint8)

        if i<=100:
            new_image= torch.cat((inputs,noisy_inputs),0)
            save_noisy_image(new_image,f"C:/Users/IICT3/PycharmProjects/Noise/output100/{j}.png")
            #print(new_image.shape)


        i=i+1

        save_noisy_image(noisy_inputs,f"C:/Users/IICT3/PycharmProjects/Noise/Noisy image/{j}.png")
        j=j+1

    sigma_values = np.array(sigma_values)
    return sigma_values

sigma_values = customFunction()

type(sigma_values)

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_folder, sigma_values, transform=None):
        self.image_paths = glob.glob(os.path.join(image_folder, "*.png"))  # Adjust this for your image format
        self.sigma_values = sigma_values
        self.transform = transform

    def __len__(self):
        return len(self.sigma_values)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        sigma = self.sigma_values[index]
        return image.to(device), sigma

# Load the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

input_dir="C:/Users/IICT3/PycharmProjects/Noise/Noisy image"
dataset = CustomDataset(image_folder=input_dir, sigma_values=sigma_values, transform=transform)

len(dataset)

trainset, testset= torch.utils.data.random_split(dataset, [35000, 15000])
trainloader = DataLoader(trainset, batch_size=128,
                                         shuffle=True)
testloader = DataLoader(testset, batch_size=128,
                                         shuffle=False)
len(trainloader)

for X,y in trainloader:
    X, y = X.to(device), y.to(device)  # Move data to GPU if available
    print(X.shape)
    print(y.shape)
    break

for X,y in testloader:
    X, y = X.to(device), y.to(device)  # Move data to GPU if available
    print(X.shape)
    print(y.shape)
    break

class NoiseNet(nn.Module):
    def __init__(self):
        super(NoiseNet, self).__init__()
        self.conv1= nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), #[1, 96, 55, 55]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2), #[1, 96, 27, 27]
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), #[1, 256, 27, 27]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2), #[1, 256, 13, 13]
        )
        self.conv3= nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), #[1, 384, 13, 13]
            nn.ReLU(),
        )
        self.conv4= nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), #[1, 384, 13, 13]
            nn.ReLU(),
        )
        self.conv5= nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), #[1, 256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2), #[1, 256, 6, 6]
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc3= nn.Sequential(
            nn.Linear(4096, 1)
        )

    def forward(self, x):
        x= self.conv1(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x= self.conv4(x)
        x= self.conv5(x)
        x= x.reshape(x.shape[0], -1) #[1, 9216]
        x= self.fc1(x)
        x= self.fc2(x)
        x= self.fc3(x)
        return x.to(device)

model = NoiseNet()
print(model)


from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

def train(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU if available
        optimizer.zero_grad()
        outputs = model(inputs)
        # Convert the target tensor to Float to match the data type of the model output
        targets = targets.float()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = running_loss / len(train_loader)
    print(f'Train Epoch: {epoch}\tAverage Loss: {avg_loss:.6f}')
    writer.add_scalar('Training Loss', avg_loss, epoch)

def test(model, test_loader, criterion, epoch, writer):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU if available
            outputs = model(inputs)
            # Convert the target tensor to Float to match the data type of the model output
            targets = targets.float()
            test_loss += criterion(outputs, targets).item()

    avg_loss = test_loss / len(test_loader)
    print(f'Test Epoch: {epoch}\tAverage Loss: {avg_loss:.6f}')
    writer.add_scalar('Testing Loss', avg_loss, epoch)

# Set up TensorBoard writer
writer = SummaryWriter()

# Assuming you've already defined your model, loss_fn, optimizer
model = NoiseNet().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and testing loop
epochs = 50
for epoch in range(1, epochs + 1):
    train(model, trainloader, loss_fn, optimizer, epoch, writer)
    test(model, testloader, loss_fn, epoch, writer)

# Close the TensorBoard writer
writer.close()
