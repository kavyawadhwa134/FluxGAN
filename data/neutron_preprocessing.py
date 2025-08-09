"""Convert neutron trajectory CSV to numpy format for LSTM-TrajGAN."""

import pandas as pd
import numpy as np
import argparse

def normalize_coordinates(coords):
    """Normalize coordinates to [-1, 1] range."""
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)
    ranges = max_vals - min_vals
    # Avoid division by zero
    ranges[ranges == 0] = 1
    normalized = 2 * (coords - min_vals) / ranges - 1
    return normalized, min_vals, max_vals

def create_sequences(trajectory, sequence_length=50):
    """Create sequences from trajectory for training."""
    sequences = []
    masks = []
    
    if len(trajectory) <= sequence_length:
        # Pad with zeros if trajectory is too short
        padded = np.zeros((sequence_length, 3))
        padded[:len(trajectory)] = trajectory
        mask = np.zeros((sequence_length, 1))
        mask[:len(trajectory)] = 1
        sequences.append(padded)
        masks.append(mask)
    else:
        # Create overlapping sequences with step size
        step_size = max(1, sequence_length // 4)  # 25% overlap
        for i in range(0, len(trajectory) - sequence_length + 1, step_size):
            seq = trajectory[i:i + sequence_length]
            mask = np.ones((sequence_length, 1))
            sequences.append(seq)
            masks.append(mask)
    
    return sequences, masks

def process_neutron_data(csv_path, sequence_length=50):
    """Process neutron trajectory data for LSTM-TrajGAN."""
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract coordinates
    coords = df[['x', 'y', 'z']].values
    
    # Normalize coordinates
    normalized_coords, min_vals, max_vals = normalize_coordinates(coords)
    
    # Create sequences
    sequences, masks = create_sequences(normalized_coords, sequence_length)
    
    # Convert to numpy arrays
    xyz_data = np.array(sequences)
    mask_data = np.array(masks)
    
    # Save normalization parameters
    norm_params = {
        'min_vals': min_vals,
        'max_vals': max_vals,
        'sequence_length': sequence_length
    }
    
    return xyz_data, mask_data, norm_params

def save_processed_data(xyz_data, mask_data, norm_params, save_path_prefix):
    """Save processed data to files."""
    
    # Convert to numpy arrays with proper shape
    xyz_array = np.array(xyz_data, dtype=np.float32)
    mask_array = np.array(mask_data, dtype=np.float32)
    
    # Save the trajectory data separately
    np.save(f"{save_path_prefix}_xyz.npy", xyz_array)
    np.save(f"{save_path_prefix}_mask.npy", mask_array)
    
    # Save normalization parameters
    np.save(f"{save_path_prefix}_norm_params.npy", norm_params)
    
    print(f"Processed data saved:")
    print(f"  - Trajectory sequences: {xyz_array.shape}")
    print(f"  - Masks: {mask_array.shape}")
    print(f"  - Files: {save_path_prefix}_xyz.npy, {save_path_prefix}_mask.npy, {save_path_prefix}_norm_params.npy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/Sheet.csv", 
                       help="Path to neutron trajectory CSV file")
    parser.add_argument("--save_prefix", type=str, default="data/neutron_train",
                       help="Prefix for saved files")
    parser.add_argument("--sequence_length", type=int, default=50,
                       help="Length of sequences for training")
    args = parser.parse_args()
    
    # Process the data
    xyz_data, mask_data, norm_params = process_neutron_data(
        args.csv_path, args.sequence_length
    )
    
    # Save processed data
    save_processed_data(xyz_data, mask_data, norm_params, args.save_prefix)