import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from neutron_model import NeutronLSTM_TrajGAN

def denormalize_coordinates(normalized_coords, norm_params):
    """Convert normalized coordinates back to original scale."""
    min_vals = norm_params['min_vals']
    max_vals = norm_params['max_vals']
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # Avoid division by zero
    
    # Reverse normalization: x_original = (x_normalized + 1) * ranges / 2 + min_vals
    original_coords = (normalized_coords + 1) * ranges / 2 + min_vals
    return original_coords

def load_trained_model(epoch=1800):
    """Load a trained neutron trajectory GAN model."""
    
    # Load normalization parameters to get sequence length
    norm_params = np.load('data/neutron_train_norm_params.npy', allow_pickle=True).item()
    sequence_length = norm_params['sequence_length']
    
    # Initialize model
    gan = NeutronLSTM_TrajGAN(latent_dim=100, sequence_length=sequence_length)
    
    # Load trained weights
    try:
        gan.load_models(epoch)
        print(f"Successfully loaded model from epoch {epoch}")
        return gan, norm_params
    except:
        print(f"Could not load model from epoch {epoch}. Make sure the model has been trained.")
        return None, norm_params

def generate_synthetic_trajectories(gan, norm_params, num_trajectories=10, seed_trajectory=None):
    """Generate synthetic neutron trajectories."""
    
    if seed_trajectory is None:
        # Create a simple seed trajectory (start at origin, move in random direction)
        sequence_length = norm_params['sequence_length']
        seed_xyz = np.zeros((1, sequence_length, 3))
        
        # Create a simple linear trajectory as seed
        for i in range(sequence_length):
            seed_xyz[0, i, 0] = i * 0.02  # x increases linearly
            seed_xyz[0, i, 1] = i * 0.01  # y increases slowly
            seed_xyz[0, i, 2] = i * 0.03  # z increases
        
        seed_mask = np.ones((1, sequence_length, 1))
    else:
        seed_xyz = seed_trajectory['xyz']
        seed_mask = seed_trajectory['mask']
    
    # Generate synthetic trajectories
    synthetic_trajectories = gan.generate_trajectories(
        num_trajectories=num_trajectories,
        seed_trajectory={'xyz': seed_xyz, 'mask': seed_mask}
    )
    
    return synthetic_trajectories

def plot_multiple_trajectories(trajectories, norm_params, title="Synthetic Neutron Trajectories", save_path=None):
    """Plot multiple 3D trajectories."""
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        # Remove padding (assuming all points are valid for synthetic trajectories)
        valid_points = traj[np.any(traj != 0, axis=1)]  # Remove zero-padded points
        
        if len(valid_points) > 1:
            # Denormalize coordinates
            original_coords = denormalize_coordinates(valid_points, norm_params)
            
            ax.plot(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2], 
                   color=colors[i], linewidth=2, alpha=0.7, label=f'Trajectory {i+1}')
            
            # Mark start and end points
            ax.scatter(original_coords[0, 0], original_coords[0, 1], original_coords[0, 2], 
                      color=colors[i], s=100, marker='o', alpha=0.8)
            ax.scatter(original_coords[-1, 0], original_coords[-1, 1], original_coords[-1, 2], 
                      color=colors[i], s=100, marker='s', alpha=0.8)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(title)
    
    if len(trajectories) <= 10:  # Only show legend if not too many trajectories
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def save_trajectories_to_csv(trajectories, norm_params, filename="synthetic_neutron_trajectories.csv"):
    """Save synthetic trajectories to CSV file."""
    
    all_trajectories = []
    
    for traj_id, traj in enumerate(trajectories):
        # Remove padding
        valid_points = traj[np.any(traj != 0, axis=1)]
        
        if len(valid_points) > 1:
            # Denormalize coordinates
            original_coords = denormalize_coordinates(valid_points, norm_params)
            
            # Create DataFrame for this trajectory
            traj_df = pd.DataFrame({
                'trajectory_id': [traj_id] * len(original_coords),
                'step': range(len(original_coords)),
                'x': original_coords[:, 0],
                'y': original_coords[:, 1],
                'z': original_coords[:, 2]
            })
            
            all_trajectories.append(traj_df)
    
    # Combine all trajectories
    if all_trajectories:
        combined_df = pd.concat(all_trajectories, ignore_index=True)
        combined_df.to_csv(filename, index=False)
        print(f"Saved {len(trajectories)} synthetic trajectories to {filename}")
        return combined_df
    else:
        print("No valid trajectories to save")
        return None

def compare_with_real_data(synthetic_trajectories, norm_params):
    """Compare synthetic trajectories with real data statistics."""
    
    # Load real data
    xyz_data = np.load('data/neutron_train_xyz.npy')
    
    # Calculate statistics for real data
    real_coords_flat = xyz_data.reshape(-1, 3)
    real_coords_original = denormalize_coordinates(real_coords_flat, norm_params)
    
    # Calculate statistics for synthetic data
    synthetic_coords_flat = np.vstack([traj for traj in synthetic_trajectories])
    synthetic_coords_original = denormalize_coordinates(synthetic_coords_flat, norm_params)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    coords = ['X', 'Y', 'Z']
    for i, coord in enumerate(coords):
        axes[i].hist(real_coords_original[:, i], bins=50, alpha=0.7, label='Real', density=True)
        axes[i].hist(synthetic_coords_original[:, i], bins=50, alpha=0.7, label='Synthetic', density=True)
        axes[i].set_xlabel(f'{coord} Position')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'{coord} Coordinate Distribution')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('neutron_results/coordinate_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for generating synthetic neutron trajectories."""
    
    # Parse command line arguments
    epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
    num_trajectories = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    print(f"Generating {num_trajectories} synthetic neutron trajectories using model from epoch {epoch}")
    
    # Create results directory
    os.makedirs('neutron_results', exist_ok=True)
    
    # Load trained model
    gan, norm_params = load_trained_model(epoch)
    
    if gan is None:
        print("Failed to load model. Please train the model first using neutron_train.py")
        return
    
    # Generate synthetic trajectories
    print("Generating synthetic trajectories...")
    synthetic_trajectories = generate_synthetic_trajectories(gan, norm_params, num_trajectories)
    
    # Plot trajectories
    plot_multiple_trajectories(
        synthetic_trajectories, 
        norm_params,
        title=f"Synthetic Neutron Trajectories (n={num_trajectories})",
        save_path=f'neutron_results/synthetic_trajectories_epoch_{epoch}.png'
    )
    
    # Save to CSV
    save_trajectories_to_csv(
        synthetic_trajectories, 
        norm_params,
        f'neutron_results/synthetic_trajectories_epoch_{epoch}.csv'
    )
    
    # Compare with real data
    print("Comparing with real data...")
    compare_with_real_data(synthetic_trajectories, norm_params)
    
    print("Synthetic trajectory generation completed!")

if __name__ == '__main__':
    main()