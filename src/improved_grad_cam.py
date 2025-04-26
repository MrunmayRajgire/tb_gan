import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import warnings

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import XRayClassifier, SimpleXRayClassifier

# Suppress specific CUDA warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")

class SimpleGradCAM:
    """
    A simpler and more robust implementation of GRAD-CAM that works better with complex models
    """
    def __init__(self, model, target_layer, use_cuda=False):
        """
        Initialize SimpleGradCAM
        
        Args:
            model (torch.nn.Module): The model to use
            target_layer (torch.nn.Module): The target layer to use for Grad-CAM
            use_cuda (bool): Whether to use CUDA for computation
        """
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Set device
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = self.model.to(self.device)
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        # Store feature maps
        self.feature_maps = output
    
    def _backward_hook(self, module, grad_in, grad_out):
        # Store gradients
        self.gradients = grad_out[0]
    
    def __call__(self, x, class_idx=None):
        """
        Generate CAM for the given class
        
        Args:
            x (torch.Tensor): Input tensor
            class_idx (int, optional): Class index to generate CAM for. If None, uses the predicted class.
            
        Returns:
            tuple: (heatmap, class_idx, score)
        """
        # Set model to eval mode
        self.model.eval()
        
        # Reset stored values
        self.feature_maps = None
        self.gradients = None
        
        # Move input to device
        x = x.to(self.device)
        
        try:
            # Forward pass
            with torch.set_grad_enabled(True):
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                
                # Get the class with highest probability if not specified
                if class_idx is None:
                    class_idx = torch.argmax(probs, dim=1).item()
                
                # Get score for the selected class
                score = probs[0, class_idx].item()
                
                # Backward pass for the selected class
                self.model.zero_grad()
                one_hot = torch.zeros_like(logits)
                one_hot[0, class_idx] = 1
                logits.backward(gradient=one_hot, retain_graph=True)
        except Exception as e:
            print(f"Error during model inference: {str(e)}")
            return None, None, 0.0
        
        # Check if feature maps and gradients were captured
        if self.feature_maps is None or self.gradients is None:
            print(f"Failed to capture feature maps or gradients. Layer: {self.target_layer}")
            return None, class_idx, score
        
        # Calculate CAM
        try:
            # Global average pooling of gradients
            pooled_gradients = torch.mean(self.gradients, dim=(2, 3))
            
            # Create a copy of feature maps to avoid modifying original
            weighted_feature_maps = self.feature_maps.clone()
            
            # Weight feature maps by gradients
            for i in range(pooled_gradients.size(0)):
                weighted_feature_maps[0, i, :, :] *= pooled_gradients[i]
            
            # Average feature maps to get heatmap
            heatmap = torch.mean(weighted_feature_maps, dim=1).squeeze().detach().cpu()
            
            # Apply ReLU to focus on features that have a positive influence
            heatmap = np.maximum(heatmap.numpy(), 0)
            
            # Normalize between 0-1
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
                
            return heatmap, class_idx, score
        except Exception as e:
            print(f"Error calculating CAM: {str(e)}")
            return None, class_idx, score
            
    def generate_visualization(self, image_path, output_path=None, alpha=0.4, colormap=cv2.COLORMAP_JET,
                             class_names=None, display_prediction=True, image_size=128):
        """
        Generate and save a Grad-CAM visualization for an image
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save visualization
            alpha (float): Transparency of heatmap overlay (0-1)
            colormap: OpenCV colormap to use (default: COLORMAP_JET)
            class_names (list, optional): Class names for display
            display_prediction (bool): Whether to display prediction info
            image_size (int): Size to resize image to
            
        Returns:
            tuple: (visualization_image, predicted_class, confidence)
        """
        # Preprocess image
        input_tensor, original_image = preprocess_image(image_path, image_size)
        input_tensor = input_tensor.to(self.device)
        
        # Generate GRAD-CAM
        print(f"Generating GRAD-CAM for {image_path}...")
        try:
            heatmap, pred_class, prob = self(input_tensor)
            
            if heatmap is None:
                print("Failed to generate heatmap")
                return None, pred_class, prob
                
            # Convert original image to numpy array
            img_np = np.array(original_image)
            
            # Resize heatmap to match original image size
            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            
            # Apply colormap to create colored heatmap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create blended image
            superimposed = heatmap_colored * alpha + img_np * (1 - alpha)
            superimposed = np.uint8(superimposed)
            
            # Create figure
            plt.figure(figsize=(15, 5))
            
            # Plot original image
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis('off')
            
            # Plot heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap_resized, cmap='jet')
            plt.title("Activation Map")
            plt.axis('off')
            
            # Plot superimposed image
            plt.subplot(1, 3, 3)
            plt.imshow(superimposed)
            
            # Set title with prediction info
            if display_prediction:
                if class_names and pred_class < len(class_names):
                    title = f"Pred: {class_names[pred_class]} ({prob:.2f})"
                else:
                    title = f"Pred: Class {pred_class} ({prob:.2f})"
                plt.title(f"Grad-CAM: {title}")
            else:
                plt.title("Grad-CAM Visualization")
                
            plt.axis('off')
            plt.tight_layout()
            
            # Save or display
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization to {output_path}")
                plt.close()
            else:
                plt.show()
                
            return superimposed, pred_class, prob
            
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            return None, None, 0.0

def preprocess_image(image_path, image_size=128):
    """
    Preprocess an image for the model
    
    Args:
        image_path (str): Path to the image
        image_size (int): Size to resize the image to
        
    Returns:
        tuple: (tensor, image)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Transform image to tensor
    tensor = transform(image).unsqueeze(0)
    
    return tensor, image

def create_visualization(image_path, model_path, model_type='resnet', output_path=None):
    """Create GRAD-CAM visualization for an image"""
    # Determine device and force CPU to avoid CUDA issues
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    if model_type.lower() == 'resnet':
        model = XRayClassifier(num_classes=2, pretrained=False)
    else:
        model = SimpleXRayClassifier(num_classes=2)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get target layer
    if model_type.lower() == 'resnet':
        # Using a better layer for feature extraction - the last conventional layer
        # in the last block that has rich semantic information
        target_layer = model.model.layer4[-1].conv2  # Conv2 in last bottleneck
        print(f"Target layer: {target_layer}")
        
        # If the above layer fails, try these alternatives:
        if target_layer is None or not isinstance(target_layer, torch.nn.Conv2d):
            print("Trying alternative layers...")
            try:
                target_layer = model.model.layer4[-1].bn3
                print(f"Using alternative layer: {target_layer}")
                if not isinstance(target_layer, torch.nn.Module):
                    raise ValueError("Invalid layer")
            except (AttributeError, ValueError):
                try:
                    # Try any convolutional layer in the last block
                    target_layer = model.model.layer4[-1].conv1
                    print(f"Using second alternative layer: {target_layer}")
                except AttributeError:
                    print("Could not find appropriate target layer")
                    return None, None, None
    else:
        target_layer = model.features[12]  # Last conv in the simple model
    
    # Print target layer info to debug
    print(f"Using target layer: {target_layer}")
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Initialize GRAD-CAM
    grad_cam = SimpleGradCAM(model, target_layer)
    
    # Generate GRAD-CAM
    print(f"Generating GRAD-CAM for {image_path}...")
    try:
        heatmap, pred_class, prob = grad_cam(input_tensor)
    except Exception as e:
        print(f"Error generating GRAD-CAM: {str(e)}")
        return None, None, None
    
    # Check if heatmap generation was successful
    if heatmap is None:
        print(f"Failed to generate heatmap for {image_path}")
        return None, pred_class, prob
    
    # Print shape info for debugging
    print(f"Heatmap shape: {heatmap.shape}")
    
    # Resize heatmap to match original image
    original_width, original_height = original_image.size
    heatmap_resized = cv2.resize(heatmap, (original_width, original_height))
    
    # Apply colormap to create visual heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert PIL image to numpy array
    original_np = np.array(original_image)
    
    # Create blended image
    blended = cv2.addWeighted(original_np, 0.7, heatmap_colored, 0.3, 0)
    
    # Create visualization figure
    class_names = {0: 'Normal', 1: 'Tuberculosis'}
    pred_label = class_names[pred_class]
    
    plt.figure(figsize=(14, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_np)
    plt.title(f"Original: {os.path.basename(image_path)}")
    plt.axis('off')
    
    # Pure heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title(f"GRAD-CAM Heatmap")
    plt.axis('off')
    
    # Blended visualization
    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title(f"GRAD-CAM: {pred_label} ({prob*100:.2f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return heatmap, pred_class, prob

def batch_process_directory(input_dir, model_path, output_dir, model_type='resnet'):
    """
    Process all images in a directory with GRAD-CAM
    
    Args:
        input_dir: Directory containing images
        model_path: Path to model weights
        output_dir: Directory to save visualizations
        model_type: Type of model ('resnet' or 'simple')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, img_path in enumerate(image_files):
        try:
            # Create output path
            output_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_gradcam.png"
            output_path = os.path.join(output_dir, output_name)
            
            # Create visualization
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            create_visualization(img_path, model_path, model_type, output_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Batch processing complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Improved GRAD-CAM for TB X-ray Classification')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'simple'], default='resnet', help='Model architecture')
    parser.add_argument('--batch', action='store_true', help='Process entire directory in batch mode')
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        # Process directory
        if not args.output:
            args.output = os.path.join('results', 'improved_gradcam')
        batch_process_directory(args.input, args.model, args.output, args.model_type)
    else:
        # Process single image
        create_visualization(args.input, args.model, args.model_type, args.output)

if __name__ == '__main__':
    main()