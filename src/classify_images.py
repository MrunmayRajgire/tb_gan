import os
import argparse
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import XRayClassifier, SimpleXRayClassifier

def load_model(model_path, model_type='resnet', device=None):
    """
    Load a trained classifier model.
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        model_type: Type of model architecture ('simple' or 'resnet')
        device: Device to run inference on (cuda or cpu)
        
    Returns:
        The loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model architecture
    if model_type.lower() == 'resnet':
        model = XRayClassifier(num_classes=2, pretrained=False).to(device)
    else:  # 'simple'
        model = SimpleXRayClassifier(num_classes=2).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    print(f"Model loaded from {model_path}")
    return model, device

def preprocess_image(image_path, image_size=128):
    """
    Preprocess an image for classification.
    
    Args:
        image_path: Path to the input image
        image_size: Size to resize the image to
        
    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def classify_single_image(model, image_path, device, image_size=128, show_image=False):
    """
    Classify a single X-ray image.
    
    Args:
        model: Loaded classifier model
        image_path: Path to the image file
        device: Device to run inference on
        image_size: Size to resize the image to
        show_image: Whether to display the image with prediction
        
    Returns:
        Predicted class label and probability
    """
    # Preprocess image
    image_tensor = preprocess_image(image_path, image_size).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    # Map class index to label
    class_labels = {0: 'Normal', 1: 'Tuberculosis'}
    predicted_label = class_labels[predicted_class]
    
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Visualize if requested
    if show_image:
        # Load and display the original image
        image = Image.open(image_path).convert('RGB')
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()
    
    return predicted_label, confidence

def classify_directory(model, input_dir, device, image_size=128, output_file=None):
    """
    Classify all X-ray images in a directory.
    
    Args:
        model: Loaded classifier model
        input_dir: Directory containing X-ray images
        device: Device to run inference on
        image_size: Size to resize images to
        output_file: Path to save classification results (optional)
        
    Returns:
        Dictionary of results
    """
    results = {}
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Classifying images"):
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path, image_size).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item() * 100
            
            # Map class index to label
            class_labels = {0: 'Normal', 1: 'Tuberculosis'}
            predicted_label = class_labels[predicted_class]
            
            # Store results
            results[image_path] = {
                'prediction': predicted_label,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results[image_path] = {
                'prediction': 'Error',
                'confidence': 0.0
            }
    
    # Count predictions
    normal_count = sum(1 for res in results.values() if res['prediction'] == 'Normal')
    tb_count = sum(1 for res in results.values() if res['prediction'] == 'Tuberculosis')
    error_count = sum(1 for res in results.values() if res['prediction'] == 'Error')
    
    print(f"\nClassification Summary:")
    print(f"  Normal: {normal_count}")
    print(f"  Tuberculosis: {tb_count}")
    print(f"  Errors: {error_count}")
    
    # Save results to file if requested
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("Image Path,Prediction,Confidence (%)\n")
            for image_path, res in results.items():
                f.write(f"{image_path},{res['prediction']},{res['confidence']:.2f}\n")
        print(f"Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Classify TB and Normal X-ray images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None, help='Path to save classification results (CSV)')
    parser.add_argument('--model_type', type=str, choices=['simple', 'resnet'], default='resnet', help='Model architecture')
    parser.add_argument('--image_size', type=int, default=128, help='Image size for processing')
    parser.add_argument('--show_image', action='store_true', help='Show image with prediction (only for single image)')
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model_path, args.model_type)
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Classify single image
        classify_single_image(model, args.input, device, args.image_size, args.show_image)
    else:
        # Classify all images in directory
        classify_directory(model, args.input, device, args.image_size, args.output)

if __name__ == "__main__":
    main()