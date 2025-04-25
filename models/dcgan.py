import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, image_size=128, features_g=64):
        """
        Generator network for DCGAN.
        
        Args:
            latent_dim: Dimension of the noise vector
            channels: Number of image channels (3 for RGB)
            image_size: Image size (height/width)
            features_g: Base feature dimension for generator
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        
        # Calculate initial feature map size
        self.init_size = image_size // 16
        
        self.latent_to_features = nn.Sequential(
            # Input: latent_dim
            nn.Linear(latent_dim, features_g * 8 * self.init_size * self.init_size),
            nn.BatchNorm1d(features_g * 8 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.features_to_image = nn.Sequential(
            # Input: (features_g*8) x init_size x init_size
            nn.Unflatten(1, (features_g * 8, self.init_size, self.init_size)),
            
            # First upsampling block
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second upsampling block
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third upsampling block
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final upsampling block
            nn.ConvTranspose2d(features_g, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range: [-1, 1]
        )

    def forward(self, z):
        """
        Forward pass of the generator.
        
        Args:
            z: Random noise tensor of shape [batch_size, latent_dim]
            
        Returns:
            Generated images of shape [batch_size, channels, image_size, image_size]
        """
        features = self.latent_to_features(z)
        img = self.features_to_image(features)
        return img
        
class Discriminator(nn.Module):
    def __init__(self, channels=3, image_size=128, features_d=64):
        """
        Discriminator network for DCGAN.
        
        Args:
            channels: Number of image channels (3 for RGB)
            image_size: Image size (height/width)
            features_d: Base feature dimension for discriminator
        """
        super(Discriminator, self).__init__()
        
        self.features = nn.Sequential(
            # Input: channels x image_size x image_size
            nn.Conv2d(channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsampling
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsampling
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsampling
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate feature map size after convolutions
        feature_size = image_size // 16  # After 4 layers with stride 2
        
        self.classifier = nn.Sequential(
            # Final convolution layer to reduce to a single value
            nn.Conv2d(features_d * 8, 1, kernel_size=feature_size, stride=1, padding=0),
            nn.Flatten(),  # Flatten to [batch_size, 1]
            nn.Sigmoid()  # Output range: [0, 1], 1 = real, 0 = fake
        )
    
    def forward(self, img):
        """
        Forward pass of the discriminator.
        
        Args:
            img: Input image tensor of shape [batch_size, channels, image_size, image_size]
            
        Returns:
            Probability that each image is real (range [0, 1])
        """
        features = self.features(img)
        output = self.classifier(features)
        return output

def weights_init(m):
    """
    Custom weights initialization for better GAN training.
    
    Args:
        m: PyTorch module
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)