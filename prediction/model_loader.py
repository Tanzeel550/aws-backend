import os
import requests
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Define the network architecture exactly as trained in model_train.py
class MNISTNeuralNetwork(nn.Module):
    def __init__(self):
        super(MNISTNeuralNetwork, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (784) -> Hidden layer (128)
        self.fc2 = nn.Linear(128, 64)       # Hidden layer (128) -> Hidden layer (64)
        self.fc3 = nn.Linear(64, 10)        # Hidden layer (64) -> Output layer (10)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

MODEL_URL = "https://mnist-455540676747-eu-north-1-an.s3.eu-north-1.amazonaws.com/best_mnist_model.pth"
# Locate model path inside backend/ directory
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "best_mnist_model.pth"
)

_model = None

def download_model():
    """
    Downloads the model state dictionary from AWS S3.
    """
    print(f"Model file not found. Downloading from S3: {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model successfully saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Error downloading the model: {e}")
        raise e

def get_model():
    """
    Singleton function to load the model into memory.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            download_model()
        
        try:
            _model = MNISTNeuralNetwork()
            # Always load model mapped to CPU
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            _model.load_state_dict(state_dict)
            _model.eval()
            print("Model successfully loaded into memory.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            _model = None
            raise e
    return _model

def preprocess_image(image_file):
    """
    Processes user uploaded images to conform to MNIST standards:
    - Grayscale conversion
    - Resize to 28x28
    - Contrast inversion (MNIST uses white digits on black background)
    - Normalized pixel ranges [0.0, 1.0]
    """
    # Open the uploaded file using PIL
    img = Image.open(image_file).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to array
    image_array = np.array(img, dtype=np.float32)
    
    # If the background is light (mean brightness > 127), invert colors to match dark background
    if np.mean(image_array) > 127:
        image_array = 255.0 - image_array
        
    # Normalize to [0, 1]
    image_array = image_array / 255.0
    
    # Reshape to batch format (1, 1, 28, 28)
    image_array = image_array.reshape(1, 1, 28, 28)
    
    # Convert to torch tensor
    return torch.from_numpy(image_array)

def predict_digit(image_file):
    """
    Performs classification on the given image file.
    Returns:
        predicted_class (int): Predicted digit (0-9)
        confidence (float): Prediction confidence score percentage
    """
    model = get_model()
    input_tensor = preprocess_image(image_file)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(output, 1).item()
        confidence = probabilities[0][predicted_class].item() * 100
        
    return predicted_class, confidence
