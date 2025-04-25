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

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import XRayClassifier, SimpleXRayClassifier

class GradCAM:
    """
    Implements Gradient-weighted Class Activation Mapping (GRAD-CAM) for CNN visualization.
    """
    def __init__(self, model, target_layer, device):
        """
        Initialize GradCAM.
        
        Args:
            model: Trained model
            target_layer: The layer to extract gradients from
            device: Device to run the model on
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Feature map and gradient storage
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.handle_forward = self.target_layer.register_forward_hook(self._forward_hook)
        self.handle_backward = self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Store the activations of the target layer."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Store the gradients of the target layer."""
        self.gradients = grad_output[0].detach()
    
    def remove_hooks(self):
        """Remove the registered hooks."""
        self.handle_forward.remove()
        self.handle_backward.remove()
        
    def generate_cam(self, image_tensor, target_class=None):
        """
        Generate the GRAD-CAM visualization.
        
        Args:
            image_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            cam: Heatmap with same dimensions as input image
            prediction: Predicted class
            confidence: Prediction confidence
        """
        # Forward pass
        image_tensor = image_tensor.to(self.device)
        self.model.zero_grad()
        output = self.model(image_tensor)
        pred_probabilities = F.softmax(output, dim=1)
        
        # Get prediction
        if target_class is None:
            predicted_class = torch.argmax(pred_probabilities, dim=1).item()
            confidence = pred_probabilities[0, predicted_class].item() * 100
        else:
            predicted_class = target_class
            confidence = pred_probabilities[0, target_class].item() * 100
        
        # Backward pass to get gradients
        output_class = output[0, predicted_class]
        output_class.backward()
        
        # Ensure gradients and activations are available
        if self.gradients is None or self.activations is None:
            print("Warning: Gradients or activations not captured. Check layer selection.")
            # Return a blank heatmap
            return np.zeros((1, 1)), predicted_class, confidence
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
        
        # Create class activation map
        batch_size, channels, height, width = self.activations.size()
        cam = torch.zeros(height, width, dtype=torch.float32, device=self.device)
        
        # Weighted sum of activation maps
        for i in range(channels):
            cam += pooled_gradients[i] * self.activations[0, i, :, :]
            
        # Apply ReLU
        cam = torch.relu(cam)
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
            
        # Resize to image size
        cam = cam.cpu().numpy()
        
        return cam, predicted_class, confidence

def get_target_layer(model, model_type):
    """
    Get the target layer for GRAD-CAM based on model type.
    
    Args:
        model: The model
        model_type: Model architecture type ('simple' or 'resnet')
        
    Returns:
        Target layer for GRAD-CAM
    """
    if model_type.lower() == 'resnet':
        # The proper way to access the final conv layer based on the model structure
        # For ResNet-50, we want to use the last convolutional layer before the fully connected layers
        # Based on the debug output, this is model.layer4[2].conv3
        return model.model.layer4[2].conv3
    else:  # 'simple'
        # For SimpleXRayClassifier, use the last convolutional layer (index 12) before max pooling
        # Based on the debug output, features.12 is the last Conv2d layer
        return model.features[12]

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
                     image_size=128, output_path=None, device=None):
    """
    Generate and visualize GRAD-CAM for a given image.
    
    Args:
        image_path: Path to the input image
        model: Trained model
        model_type: Type of model architecture ('simple' or 'resnet')
        target_class: Target class index (None for using predicted class)
        image_size: Image size for processing
        output_path: Path to save the visualization (None to display it)
        device: Device to run on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path, image_size)
    
    # Get target layer for GRAD-CAM
    target_layer = get_target_layer(model, model_type)
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer, device)
    
    # Generate GRAD-CAM
    heatmap, predicted_class, confidence = grad_cam.generate_cam(input_tensor, target_class)
    
    # Clean up hooks
    grad_cam.remove_hooks()
    
    # Class labels
    class_labels = {0: 'Normal', 1: 'Tuberculosis'}
    predicted_label = class_labels[predicted_class]
    
    # Resize heatmap to match original image size
    original_size = original_image.size
    heatmap_resized = cv2.resize(heatmap, original_size)
    
    # Convert heatmap to RGB
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
    
    # Convert PIL image to numpy array for overlay
    image_np = np.array(original_image)
    
    # Create overlay (0.7 * image + 0.3 * heatmap)
    cam_output = (0.7 * image_np + 0.3 * heatmap_rgb).astype(np.uint8)
    
    # Create figure with original image and GRAD-CAM overlay
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"Original: {os.path.basename(image_path)}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cam_output)
    plt.title(f"GRAD-CAM: {predicted_label} ({confidence:.2f}%)")
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

def batch_process_directory(input_dir, model, model_type, output_dir, image_size=128, device=None):
    """
    Process all images in a directory with GRAD-CAM.
    
    Args:
        input_dir: Directory containing images
        model: Trained model
        model_type: Model architecture type ('simple' or 'resnet')
        output_dir: Directory to save visualizations
        image_size: Image size for processing
        device: Device to run on
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
    
    # Get target layer for GRAD-CAM
    target_layer = get_target_layer(model, model_type)
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer, device)
    
    # Process each image
    results = []
    for img_path in tqdm(image_files, desc="Generating GRAD-CAM visualizations"):
        try:
            # Create output path
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, f"{os.path.splitext(rel_path)[0]}_gradcam.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Preprocess image
            input_tensor, original_image = preprocess_image(img_path, image_size)
            
            # Generate GRAD-CAM
            heatmap, predicted_class, confidence = grad_cam.generate_cam(input_tensor)
            
            # Class labels
            class_labels = {0: 'Normal', 1: 'Tuberculosis'}
            predicted_label = class_labels[predicted_class]
            
            # Resize heatmap to match original image size
            original_size = original_image.size
            heatmap_resized = cv2.resize(heatmap, original_size)
            
            # Convert heatmap to RGB
            heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
            
            # Convert PIL image to numpy array for overlay
            image_np = np.array(original_image)
            
            # Create overlay
            cam_output = (0.7 * image_np + 0.3 * heatmap_rgb).astype(np.uint8)
            
            # Create figure with original image and GRAD-CAM overlay
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title(f"Original: {os.path.basename(img_path)}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cam_output)
            plt.title(f"GRAD-CAM: {predicted_label} ({confidence:.2f}%)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            
            # Add to results
            results.append({
                'image': img_path,
                'prediction': predicted_label,
                'confidence': confidence,
                'output': output_path
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Clean up hooks
    grad_cam.remove_hooks()
    
    print(f"GRAD-CAM visualizations saved to {output_dir}")
    
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
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model_path, args.model_type, None)
    
    if args.batch_mode or os.path.isdir(args.input):
        # Process directory
        if args.output is None:
            args.output = os.path.join('./results/gradcam', os.path.basename(args.input))
        
        batch_process_directory(
            args.input, model, args.model_type, args.output, args.image_size, device
        )
    else:
        # Process single image
        visualize_gradcam(
            args.input, model, args.model_type, None, args.image_size, args.output, device
        )

if __name__ == "__main__":
    main()