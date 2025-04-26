import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

# Import pytorch-grad-cam modules
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import XRayClassifier, SimpleXRayClassifier

def get_target_layer(model, model_type):
    """
    Get the target layer for GRAD-CAM based on model type.
    
    Args:
        model: The model
        model_type: Model architecture type ('simple' or 'resnet')
        
    Returns:
        Target layer for GRAD-CAM
    """
    target_layers = []
    
    if model_type.lower() == 'resnet':
        # For ResNet models, try different layers in order of preference
        candidate_layers = [
            lambda m: m.model.layer4[-1],
            lambda m: m.model.layer4[-1].conv2,
            lambda m: m.model.layer3[-1],
            lambda m: m.model.layer2[-1],
            lambda m: m.model.layer1[-1]
        ]
    else:  # 'simple'
        # For SimpleXRayClassifier, try different layers
        candidate_layers = [
            lambda m: m.features[-1],  # Last layer in features
            lambda m: m.features[-2],  # Second to last layer
            lambda m: m.features[12] if len(m.features) > 12 else None,
            lambda m: m.features[8] if len(m.features) > 8 else None,
            lambda m: m.features[4] if len(m.features) > 4 else None
        ]
    
    # Try each candidate layer until one works
    for get_layer in candidate_layers:
        try:
            layer = get_layer(model)
            if layer is not None:
                target_layers.append(layer)
                print(f"Successfully found target layer: {layer}")
                break
        except (AttributeError, IndexError) as e:
            continue
    
    # If no layers were found, try a more desperate approach
    if not target_layers:
        print("Warning: Couldn't find standard target layers, attempting to use named modules")
        # Find any convolutional layer, starting from the end of the network
        conv_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_modules.append((name, module))
        
        # Take the last convolutional layer (closer to output)
        if conv_modules:
            name, module = conv_modules[-1]  # Get the last conv layer
            target_layers.append(module)
            print(f"Using fallback convolutional layer: {name}")
    
    if not target_layers:
        print("ERROR: Could not identify suitable target layer. Using a placeholder that may not work correctly.")
        # Last resort - try to find any module with parameters
        for name, module in model.named_modules():
            if list(module.parameters()):
                target_layers.append(module)
                print(f"Using emergency fallback layer: {name}")
                break
        
    return target_layers

def preprocess_image(image_path, image_size=128):
    """
    Preprocess an image for classification.
    
    Args:
        image_path: Path to the input image
        image_size: Size to resize the image to
        
    Returns:
        Preprocessed image tensor and original image
    """
    # Load original image for display
    original_image = Image.open(image_path).convert('RGB')
    
    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(original_image).unsqueeze(0)  # Add batch dimension
    return input_tensor, original_image

def visualize_gradcam(image_path, model, model_type='resnet', target_class=None, 
                     image_size=128, output_path=None, device=None, cam_algorithm='gradcam'):
    """
    Generate and visualize GRAD-CAM for a given image using prebuilt PyTorch modules.
    
    Args:
        image_path: Path to the input image
        model: Trained model
        model_type: Type of model architecture ('simple' or 'resnet')
        target_class: Target class index (None for using predicted class)
        image_size: Image size for processing
        output_path: Path to save the visualization (None to display it)
        device: Device to run on
        cam_algorithm: Which CAM algorithm to use ('gradcam', 'gradcam++', 'xgradcam', etc.)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure model is on the right device and in eval mode
    model = model.to(device)
    model.eval()
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path, image_size)
    input_tensor = input_tensor.to(device)
    
    # Get prediction first
    with torch.no_grad():
        pred_probabilities = model(input_tensor)
        
    if target_class is None:
        predicted_class = torch.argmax(pred_probabilities, dim=1).item()
    else:
        predicted_class = target_class
    
    confidence = pred_probabilities[0, predicted_class].item() * 100
    
    # Get target layer for GRAD-CAM
    target_layers = get_target_layer(model, model_type)
    print(f"Using target layers: {target_layers}")
    
    # If no target layers were found, we can't proceed with CAM
    if not target_layers:
        print("No target layers found. Cannot generate CAM visualization.")
        rgb_img = input_tensor[0].cpu().permute(1, 2, 0).numpy()
        rgb_img = (rgb_img * 0.5 + 0.5)  # Denormalize
        rgb_img = np.clip(rgb_img, 0, 1)  # Ensure values are in [0, 1]
        cam_image = (rgb_img * 255).astype(np.uint8)
        class_labels = {0: 'Normal', 1: 'Tuberculosis'}
        predicted_label = class_labels[predicted_class]
        
        # Create figure with just the original image and prediction
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title(f"Original: {os.path.basename(image_path)}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_img)
        plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Prediction visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return predicted_label, confidence
    
    # Select CAM algorithm
    cam_algorithms = {
        'gradcam': GradCAM,
        'hirescam': HiResCAM,
        'scorecam': ScoreCAM,
        'gradcam++': GradCAMPlusPlus,
        'xgradcam': XGradCAM,
        'eigencam': EigenCAM,
        'ablationcam': AblationCAM,
        'fullgrad': FullGrad
    }
    
    cam_algorithm_class = cam_algorithms.get(cam_algorithm.lower(), GradCAM)
    
    # Convert image tensor to numpy for visualization
    rgb_img = input_tensor[0].cpu().permute(1, 2, 0).numpy()
    rgb_img = (rgb_img * 0.5 + 0.5)  # Denormalize
    rgb_img = np.clip(rgb_img, 0, 1)  # Ensure values are in [0, 1]
    
    try:
        # Initialize the selected CAM algorithm without use_cuda parameter
        cam = cam_algorithm_class(
            model=model, 
            target_layers=target_layers
        )
        
        # Define target for CAM
        targets = [ClassifierOutputTarget(predicted_class)]
        
        # Generate CAM with proper input shape
        input_tensor_for_cam = input_tensor.clone().requires_grad_(True)
        
        # Generate CAM and handle potential errors
        grayscale_cam = cam(input_tensor=input_tensor_for_cam, targets=targets)
        
        # Explicitly check if grayscale_cam is None or empty
        if grayscale_cam is None or len(grayscale_cam) == 0 or grayscale_cam.size == 0:
            print("Warning: CAM generation returned empty result. Using fallback visualization.")
            raise ValueError("CAM generation failed - returned None or empty array")
            
        # Ensure we have a valid grayscale_cam with the right shape
        if grayscale_cam.shape[0] > 0:  
            grayscale_cam = grayscale_cam[0, :]  # Take the first image in batch
            
            # Get CAM overlay on image
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        else:
            raise ValueError("CAM generation failed - returned array with invalid shape")
            
    except Exception as e:
        print(f"Error generating CAM: {str(e)}")
        print("Falling back to original image with prediction overlay")
        # Fallback to just the original image
        cam_image = (rgb_img * 255).astype(np.uint8)
    
    # Class labels
    class_labels = {0: 'Normal', 1: 'Tuberculosis'}
    predicted_label = class_labels[predicted_class]
    
    # Create figure with original image and GRAD-CAM overlay
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"Original: {os.path.basename(image_path)}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"{cam_algorithm.upper()}: {predicted_label} ({confidence:.2f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"GRAD-CAM visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return predicted_label, confidence

def batch_process_directory(input_dir, model, model_type, output_dir, image_size=128, device=None, cam_algorithm='gradcam'):
    """
    Process all images in a directory with GRAD-CAM.
    
    Args:
        input_dir: Directory containing images
        model: Trained model
        model_type: Model architecture type ('simple' or 'resnet')
        output_dir: Directory to save visualizations
        image_size: Image size for processing
        device: Device to run on
        cam_algorithm: Which CAM algorithm to use
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    results = []
    for img_path in tqdm(image_files, desc=f"Generating {cam_algorithm.upper()} visualizations"):
        try:
            # Create output path
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, f"{os.path.splitext(rel_path)[0]}_{cam_algorithm.lower()}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Process the image
            predicted_label, confidence = visualize_gradcam(
                img_path, model, model_type, None, image_size, output_path, device, cam_algorithm
            )
            
            # Add to results
            results.append({
                'image': img_path,
                'prediction': predicted_label,
                'confidence': confidence,
                'output': output_path
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"{cam_algorithm.upper()} visualizations saved to {output_dir}")
    
    # Return results summary
    return results

def load_model(model_path, model_type='resnet', device=None):
    """
    Load a trained classifier model.
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        model_type: Type of model architecture ('simple' or 'resnet')
        device: Device to run inference on (cuda or cpu)
        
    Returns:
        The loaded model and device
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

def main():
    parser = argparse.ArgumentParser(description='GRAD-CAM for TB X-ray Images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None, help='Path to save visualization or directory')
    parser.add_argument('--model_type', type=str, choices=['simple', 'resnet'], default='resnet', help='Model architecture')
    parser.add_argument('--image_size', type=int, default=128, help='Image size for processing')
    parser.add_argument('--batch_mode', action='store_true', help='Process entire directory in batch')
    parser.add_argument('--cam_algorithm', type=str, default='gradcam', 
                        choices=['gradcam', 'gradcam++', 'xgradcam', 'scorecam', 'eigencam', 'ablationcam', 'fullgrad', 'hirescam'],
                        help='CAM algorithm to use')
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model_path, args.model_type, None)
    
    if args.batch_mode or os.path.isdir(args.input):
        # Process directory
        if args.output is None:
            args.output = os.path.join('./results/gradcam', os.path.basename(args.input))
        
        batch_process_directory(
            args.input, model, args.model_type, args.output, args.image_size, device, args.cam_algorithm
        )
    else:
        # Process single image
        visualize_gradcam(
            args.input, model, args.model_type, None, args.image_size, args.output, device, args.cam_algorithm
        )

if __name__ == "__main__":
    main()