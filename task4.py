'''
Author: Ravi Shankar Sankara Narayanan
NUID: 001568628
Date: 4/3/2024

File name: task4.py
Purpose: This file trains a neural network on the Fashion MNIST dataset using different hyperparameters and reports the best hyperparameters based on the performance metrics.
'''
# import statements
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import itertools

# Define the neural network class
class MyNetwork(nn.Module):

    # Define the layers of the network
    def __init__(self, num_conv_filters, conv_filter_size, dropout_rate):
        super(MyNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(1, num_conv_filters, conv_filter_size)
        self.conv2 = nn.Conv2d(num_conv_filters, 2*num_conv_filters, conv_filter_size)
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        
        # Calculate the size of the output feature maps after the convolution and pooling layers
        output_size = (28 - (conv_filter_size - 1)) // 2
        output_size = (output_size - (conv_filter_size - 1)) // 2

        self.fc1 = nn.Linear(2*num_conv_filters * output_size ** 2, 50)
        self.fc2 = nn.Linear(50, 10)
    
    # Define the forward pass
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to get the data loaders
def get_data_loaders():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader

# Function to train the network
def train_network(model, trainloader, num_epochs, optimizer):
    criterion = nn.CrossEntropyLoss() # Define the loss function

    # Loop through the epochs
    for epoch in range(num_epochs): 
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
# Function to evaluate the model
def evaluate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# Function to analyze the results
def analyze_results(results):
    best_accuracy = max(results, key=lambda x:x[1])
    return best_accuracy

# Function to train the model with different hyperparameters
def train_model(parameters, trainloader, testloader):
    
    # Extract the hyperparameters
    num_conv_filters = int(parameters['num_conv_filters'])
    conv_filter_size = int(parameters['conv_filter_size'])
    dropout_rate = float(parameters['dropout_rate'])
    num_epochs = int(parameters['num_epochs'])
    learning_rate = float(parameters['learning_rate'])
    
    # Initialize the model and optimizer
    model = MyNetwork(num_conv_filters, conv_filter_size, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_network(model, trainloader, num_epochs, optimizer)
    
    # Evaluate the model
    metrics = evaluate_model(model, testloader)
    
    return parameters, metrics

# Main function
def main():

    # Define the hyperparameters
    parameter_values = {
        'num_conv_filters': [16, 32],
        'conv_filter_size': [3, 5],
        'dropout_rate': [0.4, 0.7],
        'learning_rate': [0.001, 0.01]
    }

    # Define the constant values
    constant_values = {
        'num_epochs': 5
    }

    results = [] # List to store the results
    trainloader, testloader = get_data_loaders() # Get the data loaders

    # Loop through the hyperparameters to automate the training process with different hyperparameters
    for parameters in itertools.product(*parameter_values.values()):
        parameters = dict(zip(parameter_values.keys(), parameters))
        parameters.update(constant_values)
        parameters, metrics = train_model(parameters, trainloader, testloader)
        print(f"For parameters {parameters}, performance metrics are {metrics}") # Print the performance metrics
        results.append((parameters, metrics)) # Append the performance metrics to the results list
    
    # Analyze the results
    best_parameters, best_metrics = max(results, key=lambda x:x[1])
    print(f"Training completed. Best parameters: {best_parameters}, Best performance metrics: {best_metrics}")

if __name__ == "__main__":
    main() # Call the main function