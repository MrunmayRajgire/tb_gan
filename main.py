import os
import argparse
from datetime import datetime
import torch
import numpy as np

from src.train_dcgan import train_dcgan
from src.train_classifier import train_classifier

def main():
    """
    Main function to run the complete pipeline:
    1. Train DCGAN models for both TB and Normal X-ray images
    2. Generate artificial X-ray images
    3. Train a classifier using both real and generated images
    4. Evaluate the classifier performance
    """
    parser = argparse.ArgumentParser(description='TB X-ray DCGAN and Classifier Pipeline')
    
    # General parameters
    parser.add_argument('--data_dir', type=str, default='./dataset', 
                      help='Directory containing the TB_Chest_Radiography_Database dataset')
    parser.add_argument('--output_dir', type=str, default=f'./output_{datetime.now().strftime("%Y%m%d_%H%M%S")}', 
                      help='Directory to save all outputs')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed for reproducibility')
    
    # DCGAN parameters
    parser.add_argument('--gan_batch_size', type=int, default=16, 
                      help='Batch size for DCGAN training')
    parser.add_argument('--gan_epochs', type=int, default=50, 
                      help='Number of epochs for DCGAN training')
    parser.add_argument('--gan_lr', type=float, default=0.0002, 
                      help='Learning rate for DCGAN')
    parser.add_argument('--latent_dim', type=int, default=100, 
                      help='Latent dimension for DCGAN')
    parser.add_argument('--image_size', type=int, default=128, 
                      help='Image size for processing')
    parser.add_argument('--skip_gan', action='store_true', 
                      help='Skip DCGAN training and use previously generated images')
    
    # Classifier parameters
    parser.add_argument('--cls_batch_size', type=int, default=32, 
                      help='Batch size for classifier training')
    parser.add_argument('--cls_epochs', type=int, default=20, 
                      help='Number of epochs for classifier training')
    parser.add_argument('--cls_lr', type=float, default=0.001, 
                      help='Learning rate for classifier')
    parser.add_argument('--model_type', type=str, choices=['simple', 'resnet'], default='resnet', 
                      help='Type of classifier model')
    parser.add_argument('--no_generated', action='store_true', 
                      help='Train classifier without using generated images')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Train DCGAN models and generate images
    if not args.skip_gan:
        print("\n" + "="*80)
        print("Step 1: Training DCGAN models and generating artificial X-ray images")
        print("="*80)
        
        # Output directory for DCGAN
        dcgan_output_dir = os.path.join(args.output_dir, 'dcgan')
        
        # Train DCGAN for Normal X-ray images
        print("\nTraining DCGAN for Normal X-ray images...")
        train_dcgan(
            data_dir=args.data_dir,
            output_dir=dcgan_output_dir,
            class_name='Normal',
            batch_size=args.gan_batch_size,
            image_size=args.image_size,
            latent_dim=args.latent_dim,
            epochs=args.gan_epochs,
            lr=args.gan_lr
        )
        
        # Train DCGAN for TB X-ray images
        print("\nTraining DCGAN for Tuberculosis X-ray images...")
        train_dcgan(
            data_dir=args.data_dir,
            output_dir=dcgan_output_dir,
            class_name='Tuberculosis',
            batch_size=args.gan_batch_size,
            image_size=args.image_size,
            latent_dim=args.latent_dim,
            epochs=args.gan_epochs,
            lr=args.gan_lr
        )
        
        # Path to the generated images (used for classifier training)
        generated_dir = os.path.join(dcgan_output_dir, 'generated')
    else:
        print("\nSkipping DCGAN training. Using previously generated images.")
        # You need to specify where the generated images are located
        generated_dir = os.path.join(args.output_dir, 'dcgan', 'generated')
    
    # 2. Train classifier
    print("\n" + "="*80)
    print("Step 2: Training classifier for TB vs Non-TB X-ray images")
    print("="*80)
    
    # Output directory for classifier
    classifier_output_dir = os.path.join(args.output_dir, 'classifier')
    
    train_classifier(
        data_dir=args.data_dir,
        generated_dir=generated_dir,
        output_dir=classifier_output_dir,
        image_size=args.image_size,
        batch_size=args.cls_batch_size,
        epochs=args.cls_epochs,
        lr=args.cls_lr,
        use_generated=not args.no_generated,
        model_type=args.model_type,
        seed=args.seed
    )
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print(f"All outputs saved to: {args.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()