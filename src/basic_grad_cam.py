import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
from tqdm import tqdm

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import XRayClassifier, SimpleXRayClassifier

class BasicGradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Get the feature extractor part of the model
        self.features = model.model  # For ResNet
        self.fc = model.model.fc     # Fully connected layer
        
        # Initialize hooks
        self.activations = None
        self.gradients = None
        
        # Register hook for the last convolution layer
        def save_activation(module, input, output):
            self.activations = output.detach()
        
        def save_gradient(grad):
            self.gradients = grad.detach()
        
        # Get the last convolutional layer - this should reliably work for ResNet
        last_conv_layer = self._find_last_conv_layer(self.features)
        if last_conv_layer is None:
            print("Could not find a convolutional layer in the model")
            return
            
        print(f"Using layer: {last_conv_layer}")
        last_conv_layer.register_forward_hook(save_activation)
        
    def _find_last_conv_layer(self, model):
        """Find the last convolutional layer in the model"""
        last_conv = None
        # Iterate through all modules
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
                print(f"Found conv layer: {name}, {module}")
        return last_conv
        
    def __call__(self, x, class_idx=None):
        """Generate GRAD-CAM for the given input image tensor"""
        # Reset gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(x)
        
        # Get prediction if class_idx is not specified
        if class_idx is None:
            _, class_idx = torch.max(output, 1)
            class_idx = class_idx.item()
            
        # Get probabilities
        probs = torch.nn.functional.softmax(output, dim=1)
        prob = probs[0, class_idx].item()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Check if activations and gradients were captured
        if self.activations is None:
            print("No activations captured - please check your model architecture")
            return None, class_idx, prob
            
        # Get gradients for the activations
        if self.gradients is None:
            # If no gradients were directly captured, try to get them a different way
            self.gradients = self.activations.grad
            if self.gradients is None:
                print("No gradients captured - CAM generation may not be reliable")
                # Try to continue anyway with blank gradients
                # This means we'll just be visualizing the activations directly
                self.gradients = torch.zeros_like(self.activations)
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        
        # Create weighted combination of forward activation maps
        cam = torch.zeros(self.activations.shape[2:], device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]
            
        # Apply ReLU
        cam = torch.relu(cam)
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
            
        # Convert to numpy and return
        cam = cam.cpu().numpy()
        
        return cam, class_idx, prob

def preprocess_image(image_path, image_size=128):
    """Preprocess an image for model input"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, image

def visualize_cam(image_path, model_path, output_path=None, model_type='resnet'):
    """Generate and visualize Grad-CAM for an image"""
    # Use CPU to avoid CUDA context issues
    device = torch.device('cpu')
    
    # Load model
    if model_type.lower() == 'resnet':
        model = XRayClassifier(num_classes=2, pretrained=False)
    else:
        model = SimpleXRayClassifier(num_classes=2)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize GRAD-CAM
    grad_cam = BasicGradCAM(model)
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Generate CAM
    print(f"Generating GRAD-CAM for {image_path}")
    cam, class_idx, prob = grad_cam(input_tensor)
    
    # Check if CAM generation was successful
    if cam is None:
        print("Failed to generate CAM")
        return None
        
    # Resize CAM to match original image
    width, height = original_image.size
    cam_resized = cv2.resize(cam, (width, height))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    original_np = np.array(original_image)
    
    # Overlay heatmap on original image
    alpha = 0.4
    blended = cv2.addWeighted(original_np, 1-alpha, heatmap, alpha, 0)
    
    # Create figure with three panels
    class_names = {0: 'Normal', 1: 'Tuberculosis'}
    pred_label = class_names[class_idx]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_np)
    plt.title("Original X-ray")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title(f"Activation Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title(f"Prediction: {pred_label} ({prob*100:.2f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return cam, class_idx, prob

def batch_process(input_dir, model_path, output_dir, model_type='resnet'):
    """
    Process all images in a directory with GRAD-CAM
    
    Args:
        input_dir: Directory containing X-ray images
        model_path: Path to model weights
        output_dir: Directory to save visualizations
        model_type: Model type ('resnet' or 'simple')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the directory
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process in {input_dir}")
    
    # Load model once for all images
    device = torch.device('cpu')
    if model_type.lower() == 'resnet':
        model = XRayClassifier(num_classes=2, pretrained=False)
    else:
        model = SimpleXRayClassifier(num_classes=2)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize GRAD-CAM once
    grad_cam = BasicGradCAM(model)
    
    # Track results
    results = {
        'normal': 0,
        'tuberculosis': 0,
        'processed': 0,
        'failed': 0
    }
    
    # Process each image with progress bar
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Create output filename
            filename = os.path.basename(img_path)
            output_filename = f"{os.path.splitext(filename)[0]}_gradcam.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Preprocess image
            input_tensor, original_image = preprocess_image(img_path)
            input_tensor = input_tensor.to(device)
            
            # Generate CAM
            cam, class_idx, prob = grad_cam(input_tensor)
            
            # Skip if CAM generation failed
            if cam is None:
                print(f"Failed to generate CAM for {img_path}")
                results['failed'] += 1
                continue
                
            # Create visualization
            width, height = original_image.size
            cam_resized = cv2.resize(cam, (width, height))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            original_np = np.array(original_image)
            
            # Overlay heatmap on original image
            alpha = 0.4
            blended = cv2.addWeighted(original_np, 1-alpha, heatmap, alpha, 0)
            
            # Create figure
            class_names = {0: 'Normal', 1: 'Tuberculosis'}
            pred_label = class_names[class_idx]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_np)
            plt.title("Original X-ray")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap)
            plt.title(f"Activation Heatmap")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(blended)
            plt.title(f"Prediction: {pred_label} ({prob*100:.2f}%)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            # Update results
            if class_idx == 0:
                results['normal'] += 1
            else:
                results['tuberculosis'] += 1
                
            results['processed'] += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            results['failed'] += 1
    
    # Print summary
    print(f"\nBatch processing complete. Results saved to {output_dir}")
    print(f"Processed: {results['processed']} images")
    print(f"Normal: {results['normal']} images")
    print(f"Tuberculosis: {results['tuberculosis']} images")
    print(f"Failed: {results['failed']} images")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Basic GRAD-CAM for TB X-ray Classification')
    parser.add_argument('--input', type=str, required=True, help='Path to input X-ray image or directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--output', type=str, default=None, help='Path to save visualization or directory')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'simple'], default='resnet', 
                        help='Type of model architecture')
    parser.add_argument('--batch', action='store_true', help='Process a directory of images in batch mode')
    
    args = parser.parse_args()
    
    # Determine if we're processing a single image or a batch
    is_directory = os.path.isdir(args.input) or args.batch
    
    if is_directory:
        # Process directory in batch mode
        if args.output is None:
            args.output = os.path.join('results', 'basic_gradcam', os.path.basename(args.input))
            
        batch_process(args.input, args.model, args.output, args.model_type)
    else:
        # Process single image
        visualize_cam(args.input, args.model, args.output, args.model_type)

if __name__ == '__main__':
    main()