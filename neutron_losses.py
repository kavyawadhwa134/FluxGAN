import tensorflow as tf
import keras
from keras.losses import binary_crossentropy, mean_squared_error

def neutron_generator_loss(real_xyz, synthetic_xyz, mask, discriminator_output):
    """
    Custom loss function for neutron trajectory generator.
    
    Args:
        real_xyz: Real neutron trajectory coordinates
        synthetic_xyz: Generated neutron trajectory coordinates  
        mask: Mask indicating valid trajectory points
        discriminator_output: Discriminator's prediction on synthetic data
    """
    
    # Adversarial loss - fool the discriminator
    adversarial_loss = binary_crossentropy(tf.ones_like(discriminator_output), discriminator_output)
    
    # Reconstruction loss - make synthetic trajectories similar to real ones
    diff = real_xyz - synthetic_xyz
    squared_diff = tf.square(diff)
    
    # Apply mask to only consider valid trajectory points
    masked_diff = squared_diff * mask
    
    # Calculate mean squared error per trajectory
    trajectory_lengths = tf.reduce_sum(mask, axis=1, keepdims=True)
    trajectory_lengths = tf.maximum(trajectory_lengths, 1.0)  # Avoid division by zero
    
    mse_per_trajectory = tf.reduce_sum(masked_diff, axis=[1, 2], keepdims=True) / trajectory_lengths
    reconstruction_loss = tf.reduce_mean(mse_per_trajectory)
    
    # Physics-based loss - encourage smooth trajectories
    # Calculate differences between consecutive points
    consecutive_diff = synthetic_xyz[:, 1:, :] - synthetic_xyz[:, :-1, :]
    consecutive_mask = mask[:, 1:, :] * mask[:, :-1, :]  # Both points must be valid
    
    # Penalize large jumps in trajectory
    smoothness_loss = tf.reduce_mean(tf.square(consecutive_diff) * consecutive_mask)
    
    # Combine losses with weights
    total_loss = (1.0 * adversarial_loss + 
                 10.0 * reconstruction_loss + 
                 1.0 * smoothness_loss)
    
    return total_loss

def neutron_discriminator_loss(real_output, fake_output):
    """
    Discriminator loss for neutron trajectory GAN.
    """
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss