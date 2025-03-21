import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import os
import time
import json

# Define CIFAR-10 class names
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]

# Set device to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model (ResNet-18 trained on CIFAR-10)
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
model.eval()  # Set model to evaluation mode
model.to(DEVICE)

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),          # Convert from OpenCV format to PIL
    transforms.Resize((32, 32)),       # Resize to CIFAR-10 input size
    transforms.ToTensor(),             # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize (mean, std for CIFAR-10)
])

# Directory containing images
directory = "cifar10-10c"
output_json = f"{os.path.basename(directory)}.json"
results = []

# Check if directory exists
if not os.path.exists(directory):
    print(f"[ERROR] Directory '{directory}' not found!")
else:
    total_images = 0
    correct_predictions = 0
    total_time = 0.0
    
    # Process each image in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB
            image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            # Get predicted class label
            label = CIFAR10_CLASSES[predicted.item()]
            ground_truth_label = filename.split("_")[0]  # Extract label from filename
            
            # Check accuracy
            correct = 1 if label == ground_truth_label else 0
            correct_predictions += correct
            total_images += 1
            
            
            # Save result to list
            results.append({
                "image_name": filename,
                "predicted_class": label,
                "accuracy": correct,
                "processing_time": inference_time
            })
    
    # Calculate accuracy
    final_accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    
    # Save results to JSON file with total accuracy and time
    results.append({
        "total_accuracy": final_accuracy,
        "total_processing_time": total_time
    })
    
    with open(output_json, "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    # Display accuracy and total inference time
    print(f"[INFO] Model Accuracy: {final_accuracy:.2f}%")
    print(f"[INFO] Total Inference Time: {total_time} sec")
    print(f"[INFO] Results saved to {output_json}")