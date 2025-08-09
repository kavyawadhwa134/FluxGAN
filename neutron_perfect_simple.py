import tensorflow as tf
import keras
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

from keras.layers import Input, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Dropout, LayerNormalization, BatchNormalization
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

class PerfectNeutronTrajGAN():
    def __init__(self, latent_dim=100, sequence_length=50):
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        
        # Very precise optimizers
        self.d_optimizer = Adam(0.0001, 0.9)
        self.g_optimizer = Adam(0.0002, 0.9)
        
        # Store real data statistics for perfect matching
        self.real_stats = None
        
        # Build models
        self.generator = self.build_perfect_generator()
        self.discriminator = self.build_perfect_discriminator()
        
        # Compile discriminator
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.d_optimizer,
            metrics=['accuracy']
        )
        
        # Combined model
        self.discriminator.trainable = False
        
        xyz_input = Input(shape=(self.sequence_length, 3), name='xyz_input')
        mask_input = Input(shape=(self.sequence_length, 1), name='mask_input')
        noise_input = Input(shape=(self.latent_dim,), name='noise_input')
        
        synthetic_xyz = self.generator([xyz_input, mask_input, noise_input])
        validity = self.discriminator([synthetic_xyz, mask_input])
        
        self.combined = Model([xyz_input, mask_input, noise_input], [validity, synthetic_xyz])
        self.combined.compile(
            loss=[self.adversarial_loss, self.perfect_reconstruction_loss],
            loss_weights=[0.1, 0.9],  # Heavy emphasis on perfect reconstruction
            optimizer=self.g_optimizer
        )

    def adversarial_loss(self, y_true, y_pred):
        """Standard adversarial loss."""
        return keras.losses.binary_crossentropy(y_true, y_pred)

    def perfect_reconstruction_loss(self, y_true, y_pred):
        """Perfect reconstruction loss with multiple precision components."""
        
        # L2 loss (main reconstruction)
        l2_loss = K.mean(K.square(y_true - y_pred))
        
        # L1 loss (sparsity)
        l1_loss = K.mean(K.abs(y_true - y_pred))
        
        # Statistical matching losses
        true_mean = K.mean(y_true, axis=[0, 1])  # Mean across batch and sequence
        pred_mean = K.mean(y_pred, axis=[0, 1])
        mean_loss = K.mean(K.square(true_mean - pred_mean))
        
        true_std = K.std(y_true, axis=[0, 1])
        pred_std = K.std(y_pred, axis=[0, 1])
        std_loss = K.mean(K.square(true_std - pred_std))
        
        # Smoothness penalty
        pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        smoothness_loss = K.mean(K.square(pred_diff))
        
        # Combine with high weights for precision
        total_loss = (100.0 * l2_loss +
                     50.0 * l1_loss +
                     200.0 * mean_loss +
                     200.0 * std_loss +
                     10.0 * smoothness_loss)
        
        return total_loss

    def build_perfect_discriminator(self):
        """Build discriminator for perfect accuracy."""
        
        xyz_input = Input(shape=(self.sequence_length, 3), name='d_xyz_input')
        mask_input = Input(shape=(self.sequence_length, 1), name='d_mask_input')
        
        # Combine inputs
        combined = Concatenate(axis=2)([xyz_input, mask_input])
        
        # Deep LSTM layers
        lstm1 = LSTM(256, return_sequences=True, 
                    recurrent_regularizer=l2(0.0001))(combined)
        lstm1 = LayerNormalization()(lstm1)
        lstm1 = Dropout(0.2)(lstm1)
        
        lstm2 = LSTM(128, return_sequences=True,
                    recurrent_regularizer=l2(0.0001))(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        lstm2 = Dropout(0.2)(lstm2)
        
        lstm3 = LSTM(64, recurrent_regularizer=l2(0.0001))(lstm2)
        lstm3 = LayerNormalization()(lstm3)
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(lstm3)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.3)(dense2)
        
        # Output
        validity = Dense(1, activation='sigmoid')(dense2)
        
        return Model([xyz_input, mask_input], validity)

    def build_perfect_generator(self):
        """Build generator for perfect accuracy."""
        
        xyz_input = Input(shape=(self.sequence_length, 3), name='g_xyz_input')
        mask_input = Input(shape=(self.sequence_length, 1), name='g_mask_input')
        noise_input = Input(shape=(self.latent_dim,), name='g_noise_input')
        
        # Process noise
        noise_dense = Dense(256, activation='relu')(noise_input)
        noise_dense = BatchNormalization()(noise_dense)
        noise_dense = Dense(128, activation='relu')(noise_dense)
        noise_dense = BatchNormalization()(noise_dense)
        
        # Expand noise to sequence
        noise_expanded = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, self.sequence_length, 1]))(noise_dense)
        
        # Combine inputs
        combined = Concatenate(axis=2)([xyz_input, mask_input, noise_expanded])
        
        # Multi-layer LSTM
        lstm1 = LSTM(512, return_sequences=True,
                    recurrent_regularizer=l2(0.0001))(combined)
        lstm1 = LayerNormalization()(lstm1)
        lstm1 = Dropout(0.1)(lstm1)
        
        lstm2 = LSTM(256, return_sequences=True,
                    recurrent_regularizer=l2(0.0001))(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        lstm2 = Dropout(0.1)(lstm2)
        
        lstm3 = LSTM(128, return_sequences=True,
                    recurrent_regularizer=l2(0.0001))(lstm2)
        lstm3 = LayerNormalization()(lstm3)
        
        # Multi-head output for precision
        head1 = TimeDistributed(Dense(64, activation='relu'))(lstm3)
        head1 = TimeDistributed(Dense(3, activation='linear'))(head1)
        
        head2 = TimeDistributed(Dense(64, activation='relu'))(lstm3)
        head2 = TimeDistributed(Dense(3, activation='linear'))(head2)
        
        # Ensemble output
        ensemble = Lambda(lambda x: (x[0] + x[1]) / 2.0)([head1, head2])
        
        # Strong residual connection for perfect reconstruction
        residual_weight = 0.7  # Heavy weight on input
        synthetic_xyz = Lambda(lambda x: residual_weight * x[1] + (1 - residual_weight) * x[0])([ensemble, xyz_input])
        
        # Apply mask
        masked_xyz = Lambda(lambda x: x[0] * x[1])([synthetic_xyz, mask_input])
        
        return Model([xyz_input, mask_input, noise_input], masked_xyz)

    def create_perfect_data(self, xyz_data, mask_data):
        """Create perfectly augmented data."""
        
        print("Creating perfect training data...")
        
        # Store real statistics
        flat_data = xyz_data.reshape(-1, 3)
        valid_data = flat_data[~np.all(flat_data == 0, axis=1)]
        
        self.real_stats = {
            'mean': np.mean(valid_data, axis=0),
            'std': np.std(valid_data, axis=0),
            'min': np.min(valid_data, axis=0),
            'max': np.max(valid_data, axis=0)
        }
        
        # Massive data augmentation for perfect coverage
        augmented_xyz = [xyz_data]
        augmented_mask = [mask_data]
        
        # 1. Multiple noise levels
        noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        for noise in noise_levels:
            for _ in range(3):  # 3 versions per noise level
                noisy_data = xyz_data + np.random.normal(0, noise, xyz_data.shape)
                augmented_xyz.append(noisy_data)
                augmented_mask.append(mask_data)
        
        # 2. Scaling variations
        scales = [0.95, 0.98, 1.02, 1.05, 0.9, 1.1, 0.85, 1.15]
        for scale in scales:
            scaled_data = xyz_data * scale
            augmented_xyz.append(scaled_data)
            augmented_mask.append(mask_data)
        
        # 3. Statistical perturbations
        for _ in range(10):
            # Add perturbations based on real statistics
            perturbation = np.random.normal(0, self.real_stats['std'] * 0.1, xyz_data.shape)
            perturbed_data = xyz_data + perturbation
            augmented_xyz.append(perturbed_data)
            augmented_mask.append(mask_data)
        
        # 4. Perfect copies for exact reconstruction
        for _ in range(5):
            augmented_xyz.append(xyz_data.copy())
            augmented_mask.append(mask_data.copy())
        
        # Combine all
        final_xyz = np.vstack(augmented_xyz)
        final_mask = np.vstack(augmented_mask)
        
        print(f"Perfect data creation completed:")
        print(f"  Original: {xyz_data.shape[0]} trajectories")
        print(f"  Perfect augmented: {final_xyz.shape[0]} trajectories")
        print(f"  Augmentation factor: {final_xyz.shape[0] / xyz_data.shape[0]:.1f}x")
        
        return final_xyz, final_mask

    def train_perfect(self, xyz_data, mask_data, epochs=3000, batch_size=16, patience=200):
        """Train for perfect accuracy."""
        
        # Create perfect data
        xyz_perfect, mask_perfect = self.create_perfect_data(xyz_data, mask_data)
        
        print(f"\nPerfect training started:")
        print(f"  Target accuracy: 100%")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training data: {xyz_perfect.shape[0]} trajectories")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            
            epoch_d_loss = 0
            epoch_g_loss = 0
            n_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(len(xyz_perfect))
            
            for i in range(0, len(xyz_perfect), batch_size):
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) < batch_size:
                    continue
                
                batch_xyz = xyz_perfect[batch_indices]
                batch_mask = mask_perfect[batch_indices]
                
                # Perfect labels with slight smoothing
                real_labels = np.ones((len(batch_indices), 1)) * 0.95
                fake_labels = np.zeros((len(batch_indices), 1)) + 0.05
                
                # Generate synthetic
                noise = np.random.normal(0, 0.1, (len(batch_indices), self.latent_dim))
                synthetic_xyz = self.generator.predict([batch_xyz, batch_mask, noise], verbose=0)
                
                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch([batch_xyz, batch_mask], real_labels)
                d_loss_fake = self.discriminator.train_on_batch([synthetic_xyz, batch_mask], fake_labels)
                d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
                
                # Train generator multiple times for perfect reconstruction
                for _ in range(5):  # 5x generator training
                    noise = np.random.normal(0, 0.1, (len(batch_indices), self.latent_dim))
                    perfect_labels = np.ones((len(batch_indices), 1))
                    g_loss = self.combined.train_on_batch(
                        [batch_xyz, batch_mask, noise],
                        [perfect_labels, batch_xyz]  # Perfect reconstruction target
                    )
                
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss[0]
                n_batches += 1
            
            # Average losses
            avg_d_loss = epoch_d_loss / n_batches
            avg_g_loss = epoch_g_loss / n_batches
            
            # Check improvement
            if avg_g_loss < best_loss:
                best_loss = avg_g_loss
                patience_counter = 0
                self.save_perfect_models(epoch)
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 50 == 0:
                print(f"[Epoch {epoch:4d}/{epochs}] D: {avg_d_loss:.6f} | G: {avg_g_loss:.6f} | Best: {best_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
                break
        
        print("Perfect training completed!")

    def save_perfect_models(self, epoch):
        """Save perfect models."""
        self.generator.save_weights(f"neutron_perfect_params/generator_{epoch}.weights.h5")
        self.discriminator.save_weights(f"neutron_perfect_params/discriminator_{epoch}.weights.h5")

    def generate_perfect_trajectories(self, num_trajectories=50, use_real_seeds=True):
        """Generate perfect trajectories."""
        
        if use_real_seeds and self.real_stats is not None:
            # Create seeds based on real statistics
            synthetic_trajs = []
            
            for i in range(num_trajectories):
                # Create realistic seed
                seed_xyz = np.zeros((1, self.sequence_length, 3))
                seed_mask = np.ones((1, self.sequence_length, 1))
                
                # Fill with realistic values
                for j in range(self.sequence_length):
                    seed_xyz[0, j] = self.real_stats['mean'] + np.random.normal(0, self.real_stats['std'] * 0.5)
                
                # Generate with low noise for precision
                noise = np.random.normal(0, 0.05, (1, self.latent_dim))
                synthetic_traj = self.generator.predict([seed_xyz, seed_mask, noise], verbose=0)
                synthetic_trajs.append(synthetic_traj[0])
            
            return synthetic_trajs
        else:
            # Fallback method
            seed_xyz = np.random.normal(0, 0.1, (num_trajectories, self.sequence_length, 3))
            seed_mask = np.ones((num_trajectories, self.sequence_length, 1))
            noise = np.random.normal(0, 0.05, (num_trajectories, self.latent_dim))
            
            synthetic_trajs = self.generator.predict([seed_xyz, seed_mask, noise], verbose=0)
            return [traj for traj in synthetic_trajs]