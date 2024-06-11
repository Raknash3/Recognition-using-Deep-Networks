'''
Author: Ravi Shankar Sankara Narayanan
NUID: 001568628
Date: 3/30/2024

File name: task1e_main.py
Purpose: This file loads the trained model and test data and prints the predictions of the model on the test data.
'''

# import statements
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from myNetwork import MyNetwork


# Function to load the trained model
def load_model():
    model = MyNetwork()
    model.load_state_dict(torch.load('./trained_model.pth'))
    model.eval()
    return model

# Function to load the test data
def load_test_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)
    return testloader

# Function to get the predictions of the model on the test data
def get_predictions(model, images):
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    return outputs, predicted

# Function to print the predictions
def print_predictions(outputs, predicted, labels):
    for i in range(10):
        print("Output: ", np.round(outputs[i].detach().numpy(), 2))
        print("Predicted: ", predicted[i].item())
        print("Label: ", labels[i].item())
        print()

# Function to plot the predictions
def plot_predictions(images, predicted):
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap='gray', interpolation='none')
        plt.title("Predicted: {}".format(predicted[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Main function
def main():
    model = load_model()
    testloader = load_test_data()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs, predicted = get_predictions(model, images)
    print_predictions(outputs, predicted, labels)
    plot_predictions(images, predicted)

if __name__ == "__main__":
    main() # Call the main function