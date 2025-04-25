import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from scipy.linalg import sqrtm

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dcgan import Generator
from src.data_utils import XRayDataset

def load_generator(model_path, latent_dim=100, channels=3, image_size=128, features_g=64, device=None):
    """
    Load a trained DCGAN generator model.
    
    Args:
        model_path: Path to the saved generator weights (.pth file)
        latent_dim: Dimension of the latent noise vector
        channels: Number of image channels
        image_size: Size of generated images
        features_g: Base feature dimension for generator
        device: Device to load the model on
        
    Returns:
        The loaded generator model and device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize generator model
    generator = Generator(
        latent_dim=latent_dim,
        channels=channels,
        image_size=image_size,
        features_g=features_g
    ).to(device)
    
    # Load weights
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()  # Set model to evaluation mode
    
    print(f"Generator loaded from {model_path}")
    return generator, device

def generate_images(generator, num_images, latent_dim, device, batch_size=32):
    """
    Generate a set of images using the trained generator.
    
    Args:
        generator: Trained generator model
        num_images: Number of images to generate
        latent_dim: Dimension of the latent noise vector
        device: Device to generate images on
        batch_size: Batch size for generation
        
    Returns:
        Tensor of generated images
    """
    generator.eval()
    generated_images = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_images, batch_size), desc="Generating images"):
            # Determine the actual batch size for this iteration
            current_batch_size = min(batch_size, num_images - i)
            
            # Generate random noise
            z = torch.randn(current_batch_size, latent_dim, device=device)
            
            # Generate images
            fake_images = generator(z)
            
            # Detach from the computational graph and move to CPU
            generated_images.append(fake_images.detach().cpu())
    
    # Concatenate all batches
    return torch.cat(generated_images, dim=0)

def save_sample_grid(images, output_path, nrow=8):
    """
    Save a grid of sample images.
    
    Args:
        images: Tensor of images [N, C, H, W] with values in range [-1, 1]
        output_path: Path to save the output image
        nrow: Number of images per row
    """
    # Normalize from [-1, 1] to [0, 1]
    normalized = (images + 1) / 2
    
    # Make a grid and save
    grid = make_grid(normalized[:min(len(normalized), nrow*nrow)], nrow=nrow, normalize=False)
    save_image(grid, output_path)
    print(f"Sample grid saved to {output_path}")

def get_inception_features(model, images, batch_size=32, device=None):
    """
    Extract feature representations using a pre-trained model.
    
    Args:
        model: Pre-trained model for feature extraction
        images: Tensor of images [N, C, H, W] with values in range [-1, 1]
        batch_size: Batch size for feature extraction
        device: Device to use
        
    Returns:
        Numpy array of features [N, feature_dim]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    features = []
    
    # Ensure images are in [0, 1] range for model input
    images = (images + 1) / 2
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i+batch_size].to(device)
            batch_features = model(batch)
            features.append(batch_features.cpu().numpy())
            
    return np.concatenate(features, axis=0)

def calculate_fid(real_features, fake_features):
    """
    Calculate the Fréchet Inception Distance (FID) between real and fake image features.
    
    Args:
        real_features: Feature array from real images [N, feature_dim]
        fake_features: Feature array from generated images [N, feature_dim]
        
    Returns:
        FID score (lower is better)
    """
    # Reshape features if they have more than 2 dimensions
    if len(real_features.shape) > 2:
        real_features = real_features.reshape(real_features.shape[0], -1)
    if len(fake_features.shape) > 2:
        fake_features = fake_features.reshape(fake_features.shape[0], -1)
        
    # Calculate mean and covariance statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate sum of squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2)
    
    # Calculate sqrt of product of covariances
    # Handle numerical issues to ensure positive definite matrices
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid

def save_real_vs_fake_comparison(real_images, fake_images, output_path, num_samples=5):
    """
    Save side-by-side comparisons of real vs fake images.
    
    Args:
        real_images: Tensor of real images
        fake_images: Tensor of fake images
        output_path: Path to save the comparison image
        num_samples: Number of image pairs to show
    """
    # Select a subset of images
    num_samples = min(num_samples, len(real_images), len(fake_images))
    
    # Create a figure with a grid of sample comparisons
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2.5))
    
    # Normalize images from [-1, 1] to [0, 1] for display
    real_images = (real_images + 1) / 2
    fake_images = (fake_images + 1) / 2
    
    for i in range(num_samples):
        # Display real image
        axes[i, 0].imshow(real_images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Real Image")
        axes[i, 0].axis('off')
        
        # Display fake image
        axes[i, 1].imshow(fake_images[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Generated Image")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Real vs fake comparison saved to {output_path}")

def plot_histograms(real_images, fake_images, output_path):
    """
    Plot histograms comparing pixel intensity distributions of real vs fake images.
    
    Args:
        real_images: Tensor of real images
        fake_images: Tensor of fake images
        output_path: Path to save the histograms
    """
    # Convert tensors to numpy arrays
    real_np = ((real_images + 1) / 2).flatten().numpy()
    fake_np = ((fake_images + 1) / 2).flatten().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(real_np, bins=50, alpha=0.5, label='Real Images', density=True)
    plt.hist(fake_np, bins=50, alpha=0.5, label='Generated Images', density=True)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.title('Pixel Intensity Distribution: Real vs Generated')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"Histogram comparison saved to {output_path}")

def run_dcgan_validation(data_dir, model_path, output_dir, class_name, num_gen_images=100, 
                        latent_dim=100, image_size=128, batch_size=32):
    """
    Run validation on a trained DCGAN model.
    
    Args:
        data_dir: Directory containing real X-ray images
        model_path: Path to the saved generator model
        output_dir: Directory to save validation results
        class_name: Class name to evaluate ('Normal' or 'Tuberculosis')
        num_gen_images: Number of images to generate
        latent_dim: Dimension of latent space
        image_size: Size of generated images
        batch_size: Batch size for processing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load generator model
    generator, device = load_generator(
        model_path=model_path,
        latent_dim=latent_dim,
        channels=3,
        image_size=image_size,
        device=device
    )
    
    # Create data transformation for real images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Range: [-1, 1]
    ])
    
    # Create dataset for real images (using subset matching class_name)
    class_labels = {class_name: 0}  # We only need the specific class
    dataset = XRayDataset(data_dir, transform=transform, class_labels=class_labels)
    print(f"Loaded {len(dataset)} real {class_name} X-ray images")
    
    # Get a sample of real images
    real_loader = DataLoader(dataset, batch_size=num_gen_images, shuffle=True)
    real_images = next(iter(real_loader))[0]  # Get images from the batch (not labels)
    
    # Generate fake images
    fake_images = generate_images(generator, num_gen_images, latent_dim, device, batch_size)
    
    # Save sample grids
    save_sample_grid(real_images, os.path.join(output_dir, f'real_{class_name}_samples.png'))
    save_sample_grid(fake_images, os.path.join(output_dir, f'generated_{class_name}_samples.png'))
    
    # Save side-by-side comparisons
    save_real_vs_fake_comparison(
        real_images, 
        fake_images, 
        os.path.join(output_dir, f'comparison_{class_name}.png')
    )
    
    # Plot histograms of pixel intensities
    plot_histograms(
        real_images, 
        fake_images,
        os.path.join(output_dir, f'histogram_{class_name}.png')
    )
    
    # Load a feature extraction model (using ResNet-18 as it's smaller and faster than Inception)
    feature_extractor = models.resnet18(weights='DEFAULT')
    # Remove the final classification layer to get features
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    
    # Extract features from real and fake images
    print("Extracting features for FID calculation...")
    real_features = get_inception_features(feature_extractor, real_images, batch_size, device)
    fake_features = get_inception_features(feature_extractor, fake_images, batch_size, device)
    
    # Calculate FID score
    fid_score = calculate_fid(real_features, fake_features)
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, f'dcgan_metrics_{class_name}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"DCGAN Evaluation for {class_name} Class\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Number of generated images: {num_gen_images}\n")
        f.write(f"Number of real images: {len(real_images)}\n\n")
        f.write(f"FID Score: {fid_score:.4f} (lower is better)\n")
        
    print(f"All DCGAN validation results saved to {output_dir}")
    print(f"FID Score: {fid_score:.4f}")
    
    return fid_score

def main():
    parser = argparse.ArgumentParser(description='Validate DCGAN Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing X-ray images (with Normal and Tuberculosis subfolders)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained DCGAN generator model')
    parser.add_argument('--output_dir', type=str, default='results/dcgan_validation',
                        help='Directory to save validation results')
    parser.add_argument('--class_name', type=str, choices=['Normal', 'Tuberculosis'], required=True,
                        help='X-ray class to evaluate')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to generate for evaluation')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Dimension of generator latent space')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Size of generated images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Run validation
    run_dcgan_validation(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        class_name=args.class_name,
        num_gen_images=args.num_images,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()