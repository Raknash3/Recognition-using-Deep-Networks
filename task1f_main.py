'''
Author: Ravi Shankar Sankara Narayanan
NUID: 001568628
Date: 3/30/2024

File name: task1f_main.py
Purpose: This file loads the trained model and reads the images from the input_digits folder and prints the predictions of the model on the images.
'''
# import statements
import torch
from torchvision import transforms
from PIL import Image
from myNetwork import MyNetwork
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Class to load the images
class ImageLoader:
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('L')
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
             transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transformed_image = transform(image)

        # Display the image
        plt.imshow(transformed_image.squeeze(), cmap='gray')
        plt.show()

        return transformed_image

# Function to load the trained model
def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to predict the labels of the images
def predict(model, images):
    with torch.no_grad():
        outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted

# Main function
def main():
    image_folder = 'E:/MSCS/CVPR/projects/project5/data/input_digits/' 
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)] # List of paths to the images
    image_dataset = ImageLoader(image_paths) # Create an instance of the ImageLoader class
    image_loader = DataLoader(image_dataset, batch_size=len(image_dataset))  # Load all images at once
    model = load_model('trained_model.pth') # Load the trained model

    # Loop through the images and print the predictions
    for i, images in enumerate(image_loader):  
        predictions = predict(model, images) 
        for j, prediction in enumerate(predictions):
            # Print the prediction
            print(f'The prediction for the image at "{image_paths[i*len(images) + j]}" is: {prediction.item()}')

if __name__ == "__main__":
    main() # Call the main function