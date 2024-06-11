'''
Author: Ravi Shankar Sankara Narayanan
NUID: 001568628
Date: 3/4/2024

File name: ext1.py
Purpose: This file loads the trained model and predicts the digit in the frame captured from the webcam.
'''
import cv2
import torch
from torchvision import transforms
from PIL import Image
from myNetwork import MyNetwork

# Function to load the trained model
def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to predict the label of the image
def predict(model, image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.fromarray(image).convert('L')
    transformed_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(transformed_image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()


# Main function
def main():
    model = load_model('trained_model.pth') # Load the trained model
    cap = cv2.VideoCapture(0) # Open the webcam

    while True:
        ret, frame = cap.read() 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        inverted_gray = cv2.bitwise_not(gray) # Invert the grayscale image

        cv2.imshow('frame', inverted_gray) # Display the frame

        key = cv2.waitKey(1) & 0xFF # Store the result of cv2.waitKey() in a variable

        if key == ord('d'): # If 'd' is pressed
            prediction = predict(model, inverted_gray) # Predict the digit in the frame
            print(f"Predicted digit: {prediction}") # Print the prediction in the terminal
            cv2.putText(frame, str(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Display the prediction on the frame
            cv2.imshow('frame', frame) 

        if key == ord('q'): 
            break

    cap.release() # Release the webcam
    cv2.destroyAllWindows() # Close all OpenCV windows

if __name__ == "__main__":
    main() # Call the main function