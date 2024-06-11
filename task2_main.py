'''
Author: Ravi Shankar Sankara Narayanan
NUID: 001568628
Date: 3/31/2024

File name: task2_main.py
Purpose: This file loads the trained model, visualizes the filters of the first convolutional layer, and applies the filters to the first image in the MNIST dataset.

'''
# import statements
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from myNetwork import MyNetwork # Import the MyNetwork class from myNetwork.py

# Function to load the trained model
def load_model():
    model = MyNetwork()
    model.load_state_dict(torch.load('./trained_model.pth'))
    return model

# Function to get the weights of the first convolutional layer
def get_weights(model):
    weights = model.conv1.weight.detach().numpy()
    return weights

# Function to print the weights of the filters
def print_weights(weights):
    for i in range(10):
        print("Filter", i, "weights:")
        print(weights[i, 0])
        print("Shape:", weights[i, 0].shape)
        print()

# Function to visualize the filters
def visualize_filters(weights):
    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(3, 4, i+1)
        plt.imshow(weights[i, 0], cmap='gray')
        plt.title('Filter ' + str(i+1))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Function to load the first image from the MNIST dataset
def load_first_image():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    image = images[0].numpy().squeeze()
    return image

# Function to apply the filters to the image
def apply_filters(image, weights):
    filtered_images = []
    with torch.no_grad():
        for i in range(10):
            filter = weights[i, 0]
            filtered_image = cv2.filter2D(image, -1, filter)
            filtered_images.append(filtered_image)
    return filtered_images

# Function to visualize the filtered images
def visualize_filtered_images(filtered_images):
    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(3, 4, i+1)
        plt.imshow(filtered_images[i], cmap='gray')
        plt.title('Filter ' + str(i+1))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Main function
def main():
    model = load_model()
    print(model) # Print the model

    # Print the weights of the filters
    weights = get_weights(model)
    print_weights(weights) 
    visualize_filters(weights) # Visualize the filters
    
    # Visualize the filters applied to the first image 
    image = load_first_image() 
    filtered_images = apply_filters(image, weights)
    visualize_filtered_images(filtered_images)

if __name__ == "__main__":
    main() # Call the main function