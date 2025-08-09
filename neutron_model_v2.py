import tensorflow as tf
import keras
import numpy as np
import random

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

from keras.layers import Input, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Dropout, LayerNormalization
from keras.initializers import he_uniform, glorot_uniform
from keras.regularizers import l1, l2
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

class ImprovedNeutronLSTM_TrajGAN():
    def __init__(self, latent_dim=100, sequence_length=50):
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        
        # Define the optimizer with lower learning rates
        self.d_optimizer = Adam(0.0001, 0.5)  # Slower discriminator
        self.g_optimizer = Adam(0.0002, 0.5)  # Faster generator

        # Build the trajectory generator
        self.generator = self.build_generator()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy', 
            optimizer=self.d_optimizer, 
            metrics=['accuracy']
        )

        # For the combined model, we will only train the generator
        self.discriminator.trainable = False

        # Generator inputs
        xyz_input = Input(shape=(self.sequence_length, 3), name='xyz_input')
        mask_input = Input(shape=(self.sequence_length, 1), name='mask_input')
        noise_input = Input(shape=(self.latent_dim,), name='noise_input')

        # Generate synthetic trajectory
        synthetic_xyz = self.generator([xyz_input, mask_input, noise_input])

        # Discriminator prediction on synthetic trajectory
        validity = self.discriminator([synthetic_xyz, mask_input])

        # Combined model
        self.combined = Model([xyz_input, mask_input, noise_input], validity)
        self.combined.compile(
            loss=self.improved_generator_loss,
            optimizer=self.g_optimizer
        )

    def build_discriminator(self):
        """Build improved discriminator network."""
        
        xyz_input = Input(shape=(self.sequence_length, 3), name='d_xyz_input')
        mask_input = Input(shape=(self.sequence_length, 1), name='d_mask_input')
        
        # Concatenate xyz and mask
        combined_input = Concatenate(axis=2)([xyz_input, mask_input])
        
        # Add noise to discriminator inputs for stability
        noise_layer = Lambda(lambda x: x + tf.random.normal(tf.shape(x), 0, 0.01))(combined_input)
        
        # LSTM layers with layer normalization and dropout
        lstm_out = LSTM(units=128, return_sequences=True, 
                       recurrent_regularizer=l2(0.001),
                       kernel_initializer=glorot_uniform())(noise_layer)
        lstm_out = LayerNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        lstm_out = LSTM(units=64, 
                       recurrent_regularizer=l2(0.001),
                       kernel_initializer=glorot_uniform())(lstm_out)
        lstm_out = LayerNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # Dense layers with batch normalization
        dense_out = Dense(32, activation='relu')(lstm_out)
        dense_out = Dropout(0.3)(dense_out)
        validity = Dense(1, activation='sigmoid')(dense_out)

        return Model([xyz_input, mask_input], validity)

    def build_generator(self):
        """Build improved generator network."""
        
        xyz_input = Input(shape=(self.sequence_length, 3), name='g_xyz_input')
        mask_input = Input(shape=(self.sequence_length, 1), name='g_mask_input')
        noise_input = Input(shape=(self.latent_dim,), name='g_noise_input')
        
        # Expand noise to sequence length with variation
        noise_expanded = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, self.sequence_length, 1]))(noise_input)
        
        # Concatenate inputs (simplified without position encoding for now)
        combined_input = Concatenate(axis=2)([xyz_input, mask_input, noise_expanded])
        
        # LSTM layers with layer normalization
        lstm_out = LSTM(units=128, return_sequences=True, 
                       recurrent_regularizer=l2(0.001),
                       kernel_initializer=glorot_uniform())(combined_input)
        lstm_out = LayerNormalization()(lstm_out)
        
        lstm_out = LSTM(units=64, return_sequences=True, 
                       recurrent_regularizer=l2(0.001),
                       kernel_initializer=glorot_uniform())(lstm_out)
        lstm_out = LayerNormalization()(lstm_out)
        
        # Output layer - generate 3D coordinates with residual connection
        synthetic_xyz_raw = TimeDistributed(Dense(3, activation='tanh', 
                                                 kernel_initializer=glorot_uniform()))(lstm_out)
        
        # Add residual connection and scaling
        synthetic_xyz_scaled = Lambda(lambda x: x * 2.0)(synthetic_xyz_raw)  # Increase output range
        synthetic_xyz = Lambda(lambda x: x[0] + 0.3 * x[1])([xyz_input, synthetic_xyz_scaled])  # Residual
        
        # Apply mask to ensure padded positions remain zero
        masked_xyz = Lambda(lambda x: x[0] * x[1])([synthetic_xyz, mask_input])

        return Model([xyz_input, mask_input, noise_input], masked_xyz)

    def improved_generator_loss(self, y_true, y_pred):
        """Enhanced loss function for generator."""
        # Adversarial loss
        adversarial_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        return adversarial_loss

    def train_with_improved_strategy(self, xyz_data, mask_data, epochs=2000, batch_size=16, 
                                   sample_interval=100, d_train_ratio=1):
        """Train with improved GAN training strategy."""
        
        # Ground truth labels with label smoothing
        valid_smooth = np.random.uniform(0.8, 1.0, (batch_size, 1))
        fake_smooth = np.random.uniform(0.0, 0.2, (batch_size, 1))

        # Learning rate scheduling
        d_lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=50, min_lr=1e-6)
        g_lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=50, min_lr=1e-6)

        d_losses = []
        g_losses = []
        
        for epoch in range(epochs):
            
            # ---------------------
            #  Train Discriminator (with reduced frequency)
            # ---------------------
            
            if epoch % d_train_ratio == 0:
                # Select random batch
                idx = np.random.randint(0, xyz_data.shape[0], batch_size)
                real_xyz = xyz_data[idx]
                real_masks = mask_data[idx]
                
                # Add noise to real data for regularization
                noise_factor = 0.05 * max(0, 1 - epoch/1000)  # Decrease noise over time
                real_xyz_noisy = real_xyz + np.random.normal(0, noise_factor, real_xyz.shape)
                
                # Generate synthetic trajectories
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                synthetic_xyz = self.generator.predict([real_xyz, real_masks, noise], verbose=0)
                
                # Train discriminator with label smoothing
                valid_labels = np.random.uniform(0.8, 1.0, (batch_size, 1))
                fake_labels = np.random.uniform(0.0, 0.2, (batch_size, 1))
                
                d_loss_real = self.discriminator.train_on_batch([real_xyz_noisy, real_masks], valid_labels)
                d_loss_fake = self.discriminator.train_on_batch([synthetic_xyz, real_masks], fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                d_loss = [0, 0]  # Skip discriminator training
            
            # ---------------------
            #  Train Generator (multiple times per discriminator update)
            # ---------------------
            
            for _ in range(2):  # Train generator twice per discriminator update
                idx = np.random.randint(0, xyz_data.shape[0], batch_size)
                real_xyz = xyz_data[idx]
                real_masks = mask_data[idx]
                
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                valid_labels = np.ones((batch_size, 1))  # Generator wants to fool discriminator
                
                g_loss = self.combined.train_on_batch([real_xyz, real_masks, noise], valid_labels)
            
            # Store losses
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            
            # Print progress and save samples
            if epoch % sample_interval == 0:
                print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss[0]:.4f}, acc.: {d_loss[1]*100:.2f}%] [G loss: {g_loss:.4f}]")
                
                # Save model checkpoints
                self.save_models(epoch)

        return d_losses, g_losses

    def save_models(self, epoch):
        """Save model weights."""
        self.generator.save_weights(f"neutron_params_v2/generator_{epoch}.weights.h5")
        self.discriminator.save_weights(f"neutron_params_v2/discriminator_{epoch}.weights.h5")
        self.combined.save_weights(f"neutron_params_v2/combined_{epoch}.weights.h5")

    def load_models(self, epoch):
        """Load pre-trained model weights."""
        self.generator.load_weights(f"neutron_params_v2/generator_{epoch}.weights.h5")
        self.discriminator.load_weights(f"neutron_params_v2/discriminator_{epoch}.weights.h5")
        print(f"Loaded improved models from epoch {epoch}")

    def generate_trajectories(self, num_trajectories=10, seed_trajectory=None, diversity_factor=1.0):
        """Generate synthetic neutron trajectories with improved diversity."""
        
        if seed_trajectory is None:
            # Create a more diverse seed trajectory
            seed_xyz = np.random.normal(0, 0.3, (1, self.sequence_length, 3))
            seed_mask = np.ones((1, self.sequence_length, 1))
        else:
            seed_xyz = seed_trajectory['xyz']
            seed_mask = seed_trajectory['mask']
        
        # Repeat seed for batch generation
        batch_xyz = np.repeat(seed_xyz, num_trajectories, axis=0)
        batch_mask = np.repeat(seed_mask, num_trajectories, axis=0)
        
        # Generate more diverse noise
        noise = np.random.normal(0, diversity_factor, (num_trajectories, self.latent_dim))
        
        # Generate synthetic trajectories
        synthetic_trajectories = self.generator.predict([batch_xyz, batch_mask, noise])
        
        return synthetic_trajectories