import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from neutron_model import NeutronLSTM_TrajGAN

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

def plot_trajectory_comparison(real_traj, synthetic_traj, epoch, save_path="neutron_results"):
    """Plot comparison between real and synthetic trajectories."""
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D plot of real trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(real_traj[:, 0], real_traj[:, 1], real_traj[:, 2], 'b-', linewidth=2, label='Real')
    ax1.scatter(real_traj[0, 0], real_traj[0, 1], real_traj[0, 2], color='green', s=100, label='Start')
    ax1.scatter(real_traj[-1, 0], real_traj[-1, 1], real_traj[-1, 2], color='red', s=100, label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Real Neutron Trajectory')
    ax1.legend()
    
    # 3D plot of synthetic trajectory
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(synthetic_traj[:, 0], synthetic_traj[:, 1], synthetic_traj[:, 2], 'r-', linewidth=2, label='Synthetic')
    ax2.scatter(synthetic_traj[0, 0], synthetic_traj[0, 1], synthetic_traj[0, 2], color='green', s=100, label='Start')
    ax2.scatter(synthetic_traj[-1, 0], synthetic_traj[-1, 1], synthetic_traj[-1, 2], color='red', s=100, label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Synthetic Neutron Trajectory')
    ax2.legend()
    
    # Overlay comparison
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(real_traj[:, 0], real_traj[:, 1], real_traj[:, 2], 'b-', linewidth=2, label='Real', alpha=0.7)
    ax3.plot(synthetic_traj[:, 0], synthetic_traj[:, 1], synthetic_traj[:, 2], 'r--', linewidth=2, label='Synthetic', alpha=0.7)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Trajectory Comparison')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/trajectory_comparison_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_neutron_gan(epochs=2000, batch_size=16, sample_interval=200):
    """Train the neutron trajectory GAN."""
    
    # Load data
    xyz_data, mask_data, norm_params = load_neutron_data()
    sequence_length = norm_params['sequence_length']
    
    # Initialize the GAN
    gan = NeutronLSTM_TrajGAN(latent_dim=100, sequence_length=sequence_length)
    
    print("Starting training...")
    print(f"Training parameters:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sample interval: {sample_interval}")
    print(f"  - Data size: {xyz_data.shape[0]} trajectories")
    
    # Training loop
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    d_losses = []
    g_losses = []
    
    for epoch in range(epochs):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select random batch
        idx = np.random.randint(0, xyz_data.shape[0], batch_size)
        real_xyz = xyz_data[idx]
        real_masks = mask_data[idx]
        
        # Generate synthetic trajectories
        noise = np.random.normal(0, 1, (batch_size, gan.latent_dim))
        synthetic_xyz = gan.generator.predict([real_xyz, real_masks, noise], verbose=0)
        
        # Train discriminator
        d_loss_real = gan.discriminator.train_on_batch([real_xyz, real_masks], valid)
        d_loss_fake = gan.discriminator.train_on_batch([synthetic_xyz, real_masks], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        noise = np.random.normal(0, 1, (batch_size, gan.latent_dim))
        g_loss = gan.combined.train_on_batch([real_xyz, real_masks, noise], valid)
        
        # Store losses
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        
        # Print progress and save samples
        if epoch % sample_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss[0]:.4f}, acc.: {d_loss[1]*100:.2f}%] [G loss: {g_loss:.4f}]")
            
            # Generate and save sample trajectories
            sample_real = real_xyz[0]  # First trajectory in batch
            sample_synthetic = synthetic_xyz[0]  # Corresponding synthetic trajectory
            
            # Remove padding (where mask is 0)
            sample_mask = real_masks[0].flatten()
            valid_indices = sample_mask > 0.5
            
            if np.sum(valid_indices) > 1:  # Need at least 2 points for a trajectory
                sample_real_clean = sample_real[valid_indices]
                sample_synthetic_clean = sample_synthetic[valid_indices]
                
                plot_trajectory_comparison(sample_real_clean, sample_synthetic_clean, epoch)
            
            # Save model checkpoints
            gan.save_models(epoch)
    
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
    plt.savefig('neutron_results/training_losses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training completed!")
    return gan

if __name__ == '__main__':
    # Parse command line arguments
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    sample_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    
    # Train the model
    trained_gan = train_neutron_gan(epochs, batch_size, sample_interval)