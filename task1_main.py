'''
Author: Ravi Shankar Sankara Narayanan
NUID: 001568628
Date: 3/30/2024

File name: task1_main.py
Purpose: This file trains a neural network on the MNIST dataset and plots the losses and accuracies.

'''

# import statements
import sys
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from myNetwork import MyNetwork


# Function to train the network
def train_network(model, trainloader, epochs, testloader, optimizer):
    
    # Initialize the lists to store the losses and accuracies
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    examples_seen = []  
    total_examples_seen = 0 # Counter to keep track of the number of examples seen 

    # Loop through the epochs
    for e in range(epochs):
        running_loss = 0
        correct_train = 0
        total_train = 0

        # Loop through the training data
        for images, labels in trainloader:
            total_examples_seen += images.shape[0] 
            optimizer.zero_grad()
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            loss = F.nll_loss(output, labels) # Negative log likelihood loss
            loss.backward() # Backpropagation
            optimizer.step() # Update the weights
            running_loss += loss.item()
        else:
            test_loss = 0
            correct_test = 0
            total_test = 0

            # Loop through the test data
            with torch.no_grad():
                for images, labels in testloader:
                    output = model(images)
                    _, predicted = torch.max(output.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                    test_loss += F.nll_loss(output, labels)
            train_losses.append(running_loss/len(trainloader)) # Append the average training loss
            test_losses.append(test_loss/len(testloader)) # Append the average test loss
            train_accuracies.append(100 * correct_train / total_train) # Append the training accuracy
            test_accuracies.append(100 * correct_test / total_test) # Append the test accuracy
            examples_seen.append(total_examples_seen)  # Record the number of examples seen at this point
    return train_losses, test_losses, train_accuracies, test_accuracies, examples_seen


# Function to display the images
def display_images(images, labels):
    fig, axes = plt.subplots(1, 6, figsize=(10,2))
    for idx, image in enumerate(images):
        axes[idx].imshow(image.reshape(28, 28), cmap='gray')
        axes[idx].title.set_text('Label: {}'.format(labels[idx]))
        axes[idx].axis('off')
    plt.show()


# Function to plot the losses and accuracies
def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies, examples_seen):
    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss', color='green')
    plt.plot(test_losses, label='Test loss', color='orange')
    plt.title('Losses')
    plt.legend(frameon=False)
    plt.show()

    # Plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training accuracy', color='blue')
    plt.plot(test_accuracies, label='Test accuracy', color='red')
    plt.title('Accuracies')
    plt.legend(frameon=False)
    plt.show()

    # Plot the negative log likelihood loss
    plt.figure(figsize=(10, 5))
    plt.plot(examples_seen, train_losses, label='Training loss', color='green')
    plt.scatter(examples_seen, test_losses, label='Test loss', color='orange')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.title('Losses')
    plt.legend(frameon=False)
    plt.show()

# Function to load the data
def load_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    return trainset, testset

# Function to create the data loaders
def create_data_loaders(trainset, testset):
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)
    return trainloader, testloader

# Function to initialize the model and optimizer
def initialize_model_and_optimizer():
    model = MyNetwork()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer

# Main function
def main():
    trainset, testset = load_data() # Load the data
    
    # Display the first 6 images
    images = [testset[i][0] for i in range(6)] 
    labels = [testset[i][1] for i in range(6)]
    display_images(images, labels)
    
    trainloader, testloader = create_data_loaders(trainset, testset) # Create the data loaders
    model, optimizer = initialize_model_and_optimizer() # Initialize the model and optimizer

    # Train the network
    train_losses, test_losses, train_accuracies, test_accuracies, examples_seen = train_network(model, trainloader, 15, testloader, optimizer)
    torch.save(model.state_dict(), 'trained_model.pth') # Save the model
    plot_losses(train_losses, test_losses, train_accuracies, test_accuracies,examples_seen) # Plot the losses and accuracies

if __name__ == "__main__":
    main() # Call the main function