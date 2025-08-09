import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from neutron_model_v2 import ImprovedNeutronLSTM_TrajGAN

def load_neutron_data():
    """Load preprocessed neutron trajectory data."""
    xyz_data = np.load('data/neutron_train_xyz.npy')
    mask_data = np.load('data/neutron_train_mask.npy')
    norm_params = np.load('data/neutron_train_norm_params.npy', allow_pickle=True).item()
    
    print(f"Loaded neutron trajectory data:")
    print(f"  - XYZ shape: {xyz_data.shape}")
    print(f"  - Mask shape: {mask_data.shape}")
    print(f"  - Sequence length: {norm_params['sequence_length']}")
    
    return xyz_data, mask_data, norm_params

def data_augmentation(xyz_data, mask_data, augmentation_factor=3):
    """Apply data augmentation to increase dataset size."""
    
    augmented_xyz = [xyz_data]
    augmented_mask = [mask_data]
    
    for i in range(augmentation_factor):
        # Add noise augmentation
        noise_factor = 0.05 * (i + 1)
        noisy_xyz = xyz_data + np.random.normal(0, noise_factor, xyz_data.shape)
        augmented_xyz.append(noisy_xyz)
        augmented_mask.append(mask_data)
        
        # Add rotation augmentation (rotate around z-axis)
        angle = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotated_xyz = xyz_data.copy()
        # Apply 2D rotation to x,y coordinates
        rotated_xyz[:, :, 0] = xyz_data[:, :, 0] * cos_a - xyz_data[:, :, 1] * sin_a
        rotated_xyz[:, :, 1] = xyz_data[:, :, 0] * sin_a + xyz_data[:, :, 1] * cos_a
        
        augmented_xyz.append(rotated_xyz)
        augmented_mask.append(mask_data)
    
    # Combine all augmented data
    final_xyz = np.vstack(augmented_xyz)
    final_mask = np.vstack(augmented_mask)
    
    print(f"Data augmentation completed:")
    print(f"  - Original size: {xyz_data.shape[0]} trajectories")
    print(f"  - Augmented size: {final_xyz.shape[0]} trajectories")
    
    return final_xyz, final_mask

def train_improved_neutron_gan(epochs=2000, batch_size=16, sample_interval=100):
    """Train the improved neutron trajectory GAN."""
    
    # Load data
    xyz_data, mask_data, norm_params = load_neutron_data()
    sequence_length = norm_params['sequence_length']
    
    # Apply data augmentation
    print("Applying data augmentation...")
    xyz_data_aug, mask_data_aug = data_augmentation(xyz_data, mask_data, augmentation_factor=5)
    
    # Initialize the improved GAN
    gan = ImprovedNeutronLSTM_TrajGAN(latent_dim=100, sequence_length=sequence_length)
    
    print("Starting improved training...")
    print(f"Training parameters:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sample interval: {sample_interval}")
    print(f"  - Data size: {xyz_data_aug.shape[0]} trajectories")
    
    # Create results directory
    os.makedirs('neutron_params_v2', exist_ok=True)
    os.makedirs('neutron_results_v2', exist_ok=True)
    
    # Train with improved strategy
    d_losses, g_losses = gan.train_with_improved_strategy(
        xyz_data_aug, mask_data_aug, 
        epochs=epochs, 
        batch_size=batch_size, 
        sample_interval=sample_interval,
        d_train_ratio=2  # Train discriminator every 2nd epoch
    )
    
    # Plot training losses
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Training Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('neutron_results_v2/improved_training_losses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Improved training completed!")
    return gan

def generate_and_evaluate_improved(gan, norm_params, num_trajectories=20):
    """Generate trajectories and perform quick evaluation."""
    
    print(f"Generating {num_trajectories} improved synthetic trajectories...")
    
    # Generate with higher diversity
    synthetic_trajectories = gan.generate_trajectories(
        num_trajectories=num_trajectories, 
        diversity_factor=1.5
    )
    
    # Quick statistics comparison
    print("\nQuick Statistics Comparison:")
    print("-" * 50)
    
    # Load real data for comparison
    real_df = pd.read_csv('data/Sheet.csv')
    real_coords = real_df[['x', 'y', 'z']].values
    
    # Flatten synthetic data
    synthetic_coords = np.vstack([traj for traj in synthetic_trajectories])
    
    # Denormalize synthetic coordinates
    min_vals = norm_params['min_vals']
    max_vals = norm_params['max_vals']
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    synthetic_coords_original = (synthetic_coords + 1) * ranges / 2 + min_vals
    
    # Compare means and stds
    for i, coord in enumerate(['X', 'Y', 'Z']):
        real_mean = np.mean(real_coords[:, i])
        real_std = np.std(real_coords[:, i])
        synth_mean = np.mean(synthetic_coords_original[:, i])
        synth_std = np.std(synthetic_coords_original[:, i])
        
        print(f"{coord} - Real: μ={real_mean:.3f}, σ={real_std:.3f} | Synthetic: μ={synth_mean:.3f}, σ={synth_std:.3f}")
    
    return synthetic_trajectories

if __name__ == '__main__':
    # Parse command line arguments
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    sample_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # Train the improved model
    trained_gan = train_improved_neutron_gan(epochs, batch_size, sample_interval)
    
    # Quick evaluation
    import pandas as pd
    norm_params = np.load('data/neutron_train_norm_params.npy', allow_pickle=True).item()
    synthetic_trajectories = generate_and_evaluate_improved(trained_gan, norm_params)