import numpy as np
import pandas as pd
import sys
import os

from neutron_model_v2 import ImprovedNeutronLSTM_TrajGAN

def load_improved_model(epoch=450):
    """Load the improved trained model."""
    
    # Load normalization parameters
    norm_params = np.load('data/neutron_train_norm_params.npy', allow_pickle=True).item()
    sequence_length = norm_params['sequence_length']
    
    # Initialize improved model
    gan = ImprovedNeutronLSTM_TrajGAN(latent_dim=100, sequence_length=sequence_length)
    
    # Load trained weights
    try:
        gan.load_models(epoch)
        print(f"Successfully loaded improved model from epoch {epoch}")
        return gan, norm_params
    except:
        print(f"Could not load improved model from epoch {epoch}.")
        return None, norm_params

def generate_improved_trajectories(gan, norm_params, num_trajectories=20):
    """Generate synthetic trajectories using improved model."""
    
    print(f"Generating {num_trajectories} improved synthetic trajectories...")
    
    # Generate with higher diversity
    synthetic_trajectories = gan.generate_trajectories(
        num_trajectories=num_trajectories,
        diversity_factor=1.5
    )
    
    return synthetic_trajectories

def denormalize_coordinates(normalized_coords, norm_params):
    """Convert normalized coordinates back to original scale."""
    min_vals = norm_params['min_vals']
    max_vals = norm_params['max_vals']
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    
    original_coords = (normalized_coords + 1) * ranges / 2 + min_vals
    return original_coords

def save_improved_trajectories(trajectories, norm_params, filename="improved_synthetic_trajectories.csv"):
    """Save improved synthetic trajectories to CSV."""
    
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
        print(f"Saved {len(trajectories)} improved synthetic trajectories to {filename}")
        return combined_df
    else:
        print("No valid trajectories to save")
        return None

def main():
    """Generate improved synthetic trajectories."""
    
    epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 450
    num_trajectories = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    print(f"Generating {num_trajectories} improved synthetic trajectories using model from epoch {epoch}")
    
    # Load improved model
    gan, norm_params = load_improved_model(epoch)
    
    if gan is None:
        print("Failed to load improved model.")
        return
    
    # Generate synthetic trajectories
    synthetic_trajectories = generate_improved_trajectories(gan, norm_params, num_trajectories)
    
    # Save to CSV
    save_improved_trajectories(
        synthetic_trajectories,
        norm_params,
        f'neutron_results_v2/improved_synthetic_trajectories_epoch_{epoch}.csv'
    )
    
    print("Improved synthetic trajectory generation completed!")

if __name__ == '__main__':
    main()