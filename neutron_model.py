import tensorflow as tf
import keras
import numpy as np
import random

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

from keras.layers import Input, Dense, LSTM, Lambda, TimeDistributed, Concatenate
from keras.initializers import he_uniform
from keras.regularizers import l1
from keras.models import Model
from keras.optimizers import Adam

class NeutronLSTM_TrajGAN():
    def __init__(self, latent_dim=100, sequence_length=50):
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        
        # Define the optimizer
        self.optimizer = Adam(0.0002, 0.5)  # Lower learning rate for stability

        # Build the trajectory generator
        self.generator = self.build_generator()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy', 
            optimizer=self.optimizer, 
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
            loss=self.generator_loss,
            optimizer=self.optimizer
        )

    def build_discriminator(self):
        """Build discriminator network."""
        
        xyz_input = Input(shape=(self.sequence_length, 3), name='d_xyz_input')
        mask_input = Input(shape=(self.sequence_length, 1), name='d_mask_input')
        
        # Concatenate xyz and mask
        combined_input = Concatenate(axis=2)([xyz_input, mask_input])
        
        # LSTM layers
        lstm_out = LSTM(units=128, return_sequences=True, 
                       recurrent_regularizer=l1(0.01))(combined_input)
        lstm_out = LSTM(units=64, recurrent_regularizer=l1(0.01))(lstm_out)
        
        # Dense layers
        dense_out = Dense(32, activation='relu')(lstm_out)
        validity = Dense(1, activation='sigmoid')(dense_out)

        return Model([xyz_input, mask_input], validity)

    def build_generator(self):
        """Build generator network."""
        
        xyz_input = Input(shape=(self.sequence_length, 3), name='g_xyz_input')
        mask_input = Input(shape=(self.sequence_length, 1), name='g_mask_input')
        noise_input = Input(shape=(self.latent_dim,), name='g_noise_input')
        
        # Expand noise to sequence length
        noise_expanded = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, self.sequence_length, 1]))(noise_input)
        
        # Concatenate inputs
        combined_input = Concatenate(axis=2)([xyz_input, mask_input, noise_expanded])
        
        # LSTM layers
        lstm_out = LSTM(units=128, return_sequences=True, 
                       recurrent_regularizer=l1(0.01))(combined_input)
        lstm_out = LSTM(units=64, return_sequences=True, 
                       recurrent_regularizer=l1(0.01))(lstm_out)
        
        # Output layer - generate 3D coordinates
        synthetic_xyz = TimeDistributed(Dense(3, activation='tanh'))(lstm_out)
        
        # Apply mask to ensure padded positions remain zero
        masked_xyz = Lambda(lambda x: x[0] * x[1])([synthetic_xyz, mask_input])

        return Model([xyz_input, mask_input, noise_input], masked_xyz)

    def generator_loss(self, y_true, y_pred):
        """Custom loss function for generator."""
        # Adversarial loss
        adversarial_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        return adversarial_loss

    def train(self, xyz_data, mask_data, epochs=1000, batch_size=32, sample_interval=100):
        """Train the LSTM-TrajGAN model."""
        
        # Ground truth labels
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of trajectories
            idx = np.random.randint(0, xyz_data.shape[0], batch_size)
            real_xyz = xyz_data[idx]
            real_masks = mask_data[idx]
            
            # Generate synthetic trajectories
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            synthetic_xyz = self.generator.predict([real_xyz, real_masks, noise], verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch([real_xyz, real_masks], valid)
            d_loss_fake = self.discriminator.train_on_batch([synthetic_xyz, real_masks], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch([real_xyz, real_masks, noise], valid)

            # Print progress
            if epoch % sample_interval == 0:
                print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss[0]:.4f}, acc.: {d_loss[1]*100:.2f}%] [G loss: {g_loss:.4f}]")
                
                # Save model checkpoints
                self.save_models(epoch)

    def save_models(self, epoch):
        """Save model weights."""
        self.generator.save_weights(f"neutron_params/generator_{epoch}.weights.h5")
        self.discriminator.save_weights(f"neutron_params/discriminator_{epoch}.weights.h5")
        self.combined.save_weights(f"neutron_params/combined_{epoch}.weights.h5")

    def generate_trajectories(self, num_trajectories=10, seed_trajectory=None):
        """Generate synthetic neutron trajectories."""
        
        if seed_trajectory is None:
            # Create a random seed trajectory
            seed_xyz = np.random.normal(0, 0.1, (1, self.sequence_length, 3))
            seed_mask = np.ones((1, self.sequence_length, 1))
        else:
            seed_xyz = seed_trajectory['xyz']
            seed_mask = seed_trajectory['mask']
        
        # Repeat seed for batch generation
        batch_xyz = np.repeat(seed_xyz, num_trajectories, axis=0)
        batch_mask = np.repeat(seed_mask, num_trajectories, axis=0)
        
        # Generate noise
        noise = np.random.normal(0, 1, (num_trajectories, self.latent_dim))
        
        # Generate synthetic trajectories
        synthetic_trajectories = self.generator.predict([batch_xyz, batch_mask, noise])
        
        return synthetic_trajectories

    def load_models(self, epoch):
        """Load pre-trained model weights."""
        self.generator.load_weights(f"neutron_params/generator_{epoch}.weights.h5")
        self.discriminator.load_weights(f"neutron_params/discriminator_{epoch}.weights.h5")
        print(f"Loaded models from epoch {epoch}")