'''
Author: Ravi Shankar Sankara Narayanan
NUID: 001568628
Date: 4/2/2024

File name: task3_main.py
Purpose: This file loads the pre-trained model, modifies the model, trains the model on the Greek dataset, and evaluates the model on the Greek dataset. 
         It also predicts the labels of the images in the input_greek_letter folder.


'''
# import statements
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from myNetwork import MyNetwork
import matplotlib.pyplot as plt
from PIL import Image
import os

# Class to transform the Greek letters
class GreekTransform:

    # Define the augmentation techniques
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0))
        ])

    # Apply the augmentation techniques
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

# Function to load the pre-trained model
def load_model():
    model = MyNetwork()
    model.load_state_dict(torch.load('./trained_model.pth'))
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(model.fc2.in_features, 3)
    return model

# Function to get the data loader for the Greek dataset
def get_data_loader(training_set_path):
    greek_train = DataLoader(
        datasets.ImageFolder(training_set_path,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           GreekTransform(),
                                                           transforms.Normalize((0.1307,), (0.3081,)) ])),
        batch_size=10,  
        shuffle=True)
    return greek_train

# Function to train the model on the Greek dataset
def train_model(model, greek_train, epochs):
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01) # Optimizer for the fully connected layer
    criterion = nn.CrossEntropyLoss() # Loss function
    train_losses = []

    # Train the model
    for e in range(epochs):
        running_loss = 0
        for images, labels in greek_train:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss/len(greek_train))
        print("Epoch: {}/{}.. ".format(e+1, epochs), "Training Loss: {:.3f}.. ".format(running_loss/len(greek_train)))
    return train_losses

# Function to plot the training error
def plot_training_error(train_losses):
    plt.plot(train_losses, label='Training loss')
    plt.legend(frameon=False)
    plt.show()

#  Function to evaluate the model on the Greek dataset
def evaluate_model(model, greek_train):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in greek_train:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the Greek letters: %d %%' % (100 * correct / total))

# Function to predict the labels of the images in the input_greek_letter folder
def predict_labels(model, image_folder):
    transform = transforms.Compose([
        transforms.Resize((128, 128), Image.LANCZOS), # Resize the image to 128x128
        transforms.ToTensor(), # Convert the image to a tensor
        GreekTransform(), # Apply the Greek transformation
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)] # Get the paths of the images
    images = [Image.open(path).convert('RGB') for path in image_paths]
    tensors = []

    # Preprocess the images
    for image in images:
        image = image.convert('L')
        tensor = transform(image).unsqueeze(0)
        tensors.append(tensor)
    with torch.no_grad():
        outputs = [model(tensor) for tensor in tensors]
    _, predictions = torch.max(torch.cat(outputs), 1)
    for i, prediction in enumerate(predictions):
        print(f'The prediction for the image at "{image_paths[i]}" is: {prediction.item()}')

# Main function
def main():
    model = load_model() # Load the pre-trained model
    greek_train = get_data_loader('E:/MSCS/CVPR/projects/project5/data/greek_train') # Get the data loader for the Greek dataset
    train_losses = train_model(model, greek_train, 100) # Train the model on the Greek dataset
    plot_training_error(train_losses) 
    print(model)
    evaluate_model(model, greek_train)
    predict_labels(model, 'E:/MSCS/CVPR/projects/project5/data/input_greek_letter') #  Predict the labels of the images in the input_greek_letter folder


if __name__ == "__main__":
    main() # Call the main function

   