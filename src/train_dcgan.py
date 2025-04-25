import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dcgan import Generator, Discriminator, weights_init
from src.data_utils import save_generated_images

class FilteredDataset(Dataset):
    """Dataset that filters images by class."""
    
    def __init__(self, data_dir, class_name, transform=None):
        self.data_dir = os.path.join(data_dir, class_name)
        self.transform = transform
        self.image_paths = []
        
        if os.path.isdir(self.data_dir):
            for img_name in os.listdir(self.data_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(self.data_dir, img_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = torchvision.io.read_image(img_path).float() / 255.0
        
        # Convert to 3 channels if grayscale
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
            
        # Normalize to [-1, 1]
        image = (image * 2) - 1
        
        if self.transform:
            image = self.transform(image)
        
        return image

def train_dcgan(data_dir, output_dir, class_name, image_size=128, batch_size=16, 
                latent_dim=100, epochs=50, lr=0.0002, beta1=0.5, beta2=0.999,
                device=None, save_interval=10, sample_size=16):
    """
    Train DCGAN model on X-ray images.
    
    Args:
        data_dir: Directory containing TB_Chest_Radiography_Database dataset
        output_dir: Directory to save model and generated images
        class_name: Class to generate ('Normal' or 'Tuberculosis')
        image_size: Size to resize images to
        batch_size: Batch size for training
        latent_dim: Dimension of the latent space (noise vector)
        epochs: Number of training epochs
        lr: Learning rate
        beta1: Beta1 parameter for Adam optimizer
        beta2: Beta2 parameter for Adam optimizer
        device: Device to use (cuda or cpu)
        save_interval: Epoch interval to save models and generate samples
        sample_size: Number of images to generate for visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, 'samples', class_name)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create class-specific dataset directly
    dataset_path = os.path.join(data_dir, 'TB_Chest_Radiography_Database')
    dataset = FilteredDataset(dataset_path, class_name, transform)
    
    # Create dataloader with a fixed batch size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True  # Ensure all batches have the same size
    )
    
    print(f"Created dataloader with {len(dataset)} {class_name} images")
    
    if len(dataloader) == 0:
        raise ValueError(f"No {class_name} images found in the dataset")
    
    # Initialize models
    netG = Generator(latent_dim=latent_dim, channels=3, image_size=image_size).to(device)
    netD = Discriminator(channels=3, image_size=image_size).to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(sample_size, latent_dim, device=device)
    
    # Labels for real and fake data
    real_label = 1.0
    fake_label = 0.0
    
    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Lists to track progress
    img_list = []
    G_losses = []
    D_losses = []
    
    print(f"Starting training on {class_name} X-ray images...")
    
    # Training loop
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, real_images in progress_bar:
            # Skip batches that are too small (this shouldn't happen with drop_last=True)
            if real_images.size(0) < 2:
                continue
                
            ############################
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with real data
            netD.zero_grad()
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with fake data
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################
            # (2) Update Generator: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # Generator wants discriminator to think its output is real
            
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optimizerG.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': errD.item(),
                'G_loss': errG.item(),
                'D(x)': D_x,
                'D(G(z))': D_G_z2
            })
            
            # Save losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
        # Save models and generate samples
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            # Save models
            torch.save(netG.state_dict(), os.path.join(model_dir, f'generator_{class_name}_{epoch+1}.pth'))
            torch.save(netD.state_dict(), os.path.join(model_dir, f'discriminator_{class_name}_{epoch+1}.pth'))
            
            # Generate and save images
            with torch.no_grad():
                netG.eval()
                fake = netG(fixed_noise).detach().cpu()
                netG.train()
            
            # Save generated images
            save_path = os.path.join(samples_dir, f'epoch_{epoch+1}')
            save_generated_images(fake, save_path, prefix=f'{class_name.lower()}')
            
            # Save a grid of images for visualization
            grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)
            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title(f"Generated {class_name} X-rays - Epoch {epoch+1}")
            plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
            plt.savefig(os.path.join(samples_dir, f'grid_epoch_{epoch+1}.png'))
            plt.close()
            
            # Add to image list
            img_list.append(grid)
    
    # After training, generate a batch of images
    num_gen_images = 100  # Number of images to generate
    noise = torch.randn(num_gen_images, latent_dim, device=device)
    with torch.no_grad():
        netG.eval()
        generated_images = netG(noise).detach().cpu()
    
    # Save the generated images
    output_gen_dir = os.path.join(output_dir, 'generated', class_name)
    os.makedirs(output_gen_dir, exist_ok=True)
    save_generated_images(generated_images, output_gen_dir, prefix=f'{class_name.lower()}')
    
    print(f"Training complete. Generated images saved to {output_gen_dir}")
    
    # Plot training losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'loss_plot_{class_name}.png'))
    plt.close()
    
    return netG, netD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DCGAN on X-ray images')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./data/generated', help='Directory to save output')
    parser.add_argument('--class_name', type=str, choices=['Normal', 'Tuberculosis'], required=True, help='Class to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train_dcgan(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        class_name=args.class_name,
        batch_size=args.batch_size,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr
    )