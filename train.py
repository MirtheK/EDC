"""
Example training script for GANomaly 3D medical image anomaly detection.

This script demonstrates how to:
1. Set up the dataset and dataloaders
2. Initialize the model
3. Configure the trainer
4. Run training
"""

import torch
from torch.utils.data import DataLoader
import argparse

# Import your modules
from models.gan.torch_model import GanomalyModel
from models.gan.loss import GeneratorLoss, DiscriminatorLoss
from datasets.dataset import AD_Dataset
from models.gan.trainer import GanomalyTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GANomaly for 3D medical image anomaly detection')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--train_samples_limit', type=int, default=10000,
                        help='Maximum number of training samples (default: 10000)')
    
    # Model arguments
    parser.add_argument('--input_size', type=int, nargs=3, default=[128, 128, 128],
                        help='Input size (D H W) (default: 128 64 128)')
    parser.add_argument('--n_features', type=int, default=64,
                        help='Number of features in CNN (default: 64)')
    parser.add_argument('--latent_vec_size', type=int, default=100,
                        help='Size of latent vector (default: 100)')
    parser.add_argument('--extra_layers', type=int, default=0,
                        help='Number of extra layers (default: 0)')
    
    # Loss weights
    parser.add_argument('--wadv', type=int, default=1,
                        help='Weight for adversarial loss (default: 1)')
    parser.add_argument('--wcon', type=int, default=50,
                        help='Weight for contextual loss (default: 50)')
    parser.add_argument('--wenc', type=int, default=1,
                        help='Weight for encoding loss (default: 1)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, required=True,
                        help='Number of training epochs')
    parser.add_argument('--val_every', type=int, default=5,
                        help='Run validation every N epochs (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1 (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2 (default: 0.999)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from (optional)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use (auto-detect if not specified)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("GANomaly 3D Medical Image Anomaly Detection Training")

    # Print configuration
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # ==================== Data Setup ====================
    print("Setting up datasets...")
    
    # Training dataset
    train_dataset_wrapper = AD_Dataset(
        name='medical-3d',
        train=True,
        data_dir=args.data_dir,
        train_samples_limit=args.train_samples_limit,
        imagenet_norm=False
    )
    train_dataset = train_dataset_wrapper.get_dset()
    
    # Validation dataset
    val_dataset_wrapper = AD_Dataset(
        name='medical-3d',
        train=False,
        data_dir=args.data_dir,
        imagenet_norm=False
    )
    val_dataset = val_dataset_wrapper.get_dset()
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # ==================== Model Setup ====================
    print("\nInitializing model...")
    
    input_size = tuple(args.input_size)  # (D, H, W)
    
    model = GanomalyModel(
        input_size=input_size,
        num_input_channels=1,  # Medical images are typically single channel
        n_features=args.n_features,
        latent_vec_size=args.latent_vec_size,
        extra_layers=args.extra_layers,
        add_final_conv_layer=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== Loss Functions ====================
    generator_loss = GeneratorLoss(
        wadv=args.wadv,
        wcon=args.wcon,
        wenc=args.wenc
    )
    
    discriminator_loss = DiscriminatorLoss()
    
    # ==================== Trainer Setup ====================
    print("\nSetting up trainer...")
    
    config = {
        'learning_rate': args.learning_rate,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'checkpoint_dir': args.checkpoint_dir,
        'device': args.device
    }
    
    trainer = GanomalyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        **config
    )
    
    # ==================== Training ====================
    print("\nStarting training...\n")
    
    trainer.train(
        num_epochs=args.num_epochs,
        val_every=args.val_every,
        resume_from=args.resume_from
    )
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pth")
    print(f"Best validation AUROC: {trainer.best_val_auroc:.4f}")


if __name__ == '__main__':
    main()