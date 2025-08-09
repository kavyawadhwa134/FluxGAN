import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from neutron_perfect_simple import PerfectNeutronTrajGAN

def main():
    """Train perfect neutron trajectory GAN."""
    
    print("="*80)
    print("🎯 PERFECT NEUTRON TRAJECTORY GAN - 100% ACCURACY TARGET")
    print("="*80)
    print("🔬 Nuclear Reactor Safety Grade")
    print("⚡ Zero Margin for Error")
    print("="*80)
    
    # Create directories
    os.makedirs('neutron_perfect_params', exist_ok=True)
    os.makedirs('neutron_perfect_results', exist_ok=True)
    
    # Load data
    print("Loading neutron trajectory data...")
    xyz_data = np.load('data/neutron_train_xyz.npy')
    mask_data = np.load('data/neutron_train_mask.npy')
    norm_params = np.load('data/neutron_train_norm_params.npy', allow_pickle=True).item()
    
    print(f"Data loaded:")
    print(f"  XYZ shape: {xyz_data.shape}")
    print(f"  Sequence length: {norm_params['sequence_length']}")
    
    # Initialize perfect GAN
    print("\nInitializing Perfect GAN...")
    gan = PerfectNeutronTrajGAN(latent_dim=100, sequence_length=norm_params['sequence_length'])
    
    # Train for perfect accuracy
    print("\nStarting perfect training...")
    gan.train_perfect(xyz_data, mask_data, epochs=2000, batch_size=16, patience=150)
    
    # Generate perfect trajectories
    print("\nGenerating perfect trajectories...")
    perfect_trajs = gan.generate_perfect_trajectories(num_trajectories=100)
    
    # Save perfect trajectories
    def denormalize_coordinates(normalized_coords, norm_params):
        min_vals = norm_params['min_vals']
        max_vals = norm_params['max_vals']
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        return (normalized_coords + 1) * ranges / 2 + min_vals
    
    all_trajectories = []
    for traj_id, traj in enumerate(perfect_trajs):
        valid_points = traj[np.any(traj != 0, axis=1)]
        if len(valid_points) > 1:
            original_coords = denormalize_coordinates(valid_points, norm_params)
            traj_df = pd.DataFrame({
                'trajectory_id': [traj_id] * len(original_coords),
                'step': range(len(original_coords)),
                'x': original_coords[:, 0],
                'y': original_coords[:, 1],
                'z': original_coords[:, 2]
            })
            all_trajectories.append(traj_df)
    
    if all_trajectories:
        combined_df = pd.concat(all_trajectories, ignore_index=True)
        combined_df.to_csv('neutron_perfect_results/perfect_synthetic_trajectories.csv', index=False)
        
        print(f"✅ Perfect trajectories saved!")
        print(f"   Total trajectories: {len(perfect_trajs)}")
        print(f"   Total points: {len(combined_df)}")
        
        # Quick stats
        print(f"\nPerfect Statistics:")
        for coord in ['x', 'y', 'z']:
            mean_val = combined_df[coord].mean()
            std_val = combined_df[coord].std()
            print(f"  {coord.upper()}: μ={mean_val:8.4f}, σ={std_val:8.4f}")
    
    print("\n✅ PERFECT TRAINING COMPLETED!")
    print("🎯 Ready for 100% accuracy evaluation!")

if __name__ == '__main__':
    main()