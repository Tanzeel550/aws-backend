import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import os
import time
from sklearn.model_selection import train_test_split

# Set device
device = torch.device('cpu')
print(f"Using device: {device}")

# Define the neural network architecture
class MNISTNeuralNetwork(nn.Module):
    def __init__(self):
        super(MNISTNeuralNetwork, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (784) -> Hidden layer (128)
        self.fc2 = nn.Linear(128, 64)       # Hidden layer (128) -> Hidden layer (64)
        self.fc3 = nn.Linear(64, 10)        # Hidden layer (64) -> Output layer (10)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)      # Dropout for regularization
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Alternative CNN architecture (better performance)
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Custom Dataset class
class MNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform
        
        # Separate features and labels
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype(np.float32)
        
        # Normalize pixel values to [0, 1]
        self.images = self.images / 255.0
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """
    Train the neural network model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model_state = None
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'best_mnist_model.pth')
            print(f"Epoch {epoch+1}: New best model saved with validation accuracy: {val_accuracy:.2f}%")
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Load best model
    model.load_state_dict(torch.load('best_mnist_model.pth', map_location=device))
    return model, best_val_accuracy

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def preprocess_image_for_mnist(image_path):
    """
    Preprocess an input image to match MNIST format:
    - Convert to grayscale
    - Resize to 28x28
    - Normalize pixel values
    - Invert colors if necessary (MNIST has white digits on black background)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Convert to numpy array and normalize
    image_array = resized.astype(np.float32)
    
    # Check if image is dark on light background (like regular writing)
    # If so, invert it to match MNIST format (white on black)
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
    
    # Normalize to [0, 1]
    image_array = image_array / 255.0
    
    # Reshape for model input (1, 28, 28)
    image_array = image_array.reshape(1, 1, 28, 28)
    
    return image_array

def test_image(model, image_path):
    """
    Test a single image using the trained model
    """
    # Preprocess the image
    image_array = preprocess_image_for_mnist(image_path)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_array).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(output, 1).item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    # Display the image and prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(image_array.squeeze(), cmap='gray')
    plt.title(f'Predicted: {predicted_class} (Confidence: {confidence:.2f}%)')
    plt.axis('off')
    plt.show()
    
    print(f"Predicted digit: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    return predicted_class, confidence

def main():
    """Main function to train and test the model"""
    
    # Check if CSV files exist
    train_file = 'mnist_train.csv'
    test_file = 'mnist_test.csv'
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Error: MNIST CSV files not found!")
        print("Make sure 'mnist_train.csv' and 'mnist_test.csv' are in the current directory.")
        return
    
    # Load datasets
    print("Loading datasets...")
    full_dataset = MNISTDataset(train_file)
    test_dataset = MNISTDataset(test_file)
    
    # Split training data into train and validation (80-20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Choose model architecture
    print("\nChoose model architecture:")
    print("1. Fully Connected Neural Network")
    print("2. Convolutional Neural Network (Recommended)")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        model = MNISTNeuralNetwork().to(device)
        print("Using Fully Connected Neural Network")
    else:
        model = MNISTCNN().to(device)
        print("Using Convolutional Neural Network")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train the model
    model, best_accuracy = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=10, 
        learning_rate=0.001
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Test with a single image
    print("\n" + "="*50)
    print("Testing with custom images")
    print("="*50)
    
    while True:
        image_path = input("\nEnter path to test image (or 'quit' to exit): ")
        if image_path.lower() == 'quit':
            break
        
        if os.path.exists(image_path):
            try:
                predicted, confidence = test_image(model, image_path)
            except Exception as e:
                print(f"Error processing image: {e}")
        else:
            print("File not found! Please enter a valid path.")

if __name__ == "__main__":
    main()
