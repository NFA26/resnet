import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import os
import time
import json

# Define CIFAR-100 class names
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

# Set device to GPU if available
DEVICE = torch.device("cpu")

# Load pre-trained model (ResNet-44 trained on CIFAR-100)
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
model.eval()  # Set model to evaluation mode
model.to(DEVICE)

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),          # Convert from OpenCV format to PIL
    transforms.Resize((32, 32)),       # Resize to CIFAR-100 input size
    transforms.ToTensor(),             # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize (mean, std for CIFAR-100)
])

# Directory containing images
directory = "/home/student/cs550project/cifar100-10c"
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
            label = CIFAR100_CLASSES[predicted.item()]
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
    print(f"[INFO] Results saved to {output_json}")
 