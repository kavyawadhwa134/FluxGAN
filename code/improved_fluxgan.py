import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils.spectral_norm as spectral_norm

# Configuration
checkpoint_dir = './plots/checkpoint'
loss_log_file = './plots/improved_loss_log.csv'
checkpoint_interval = 500
num_epochs = 10000
batch_size = 256
noise_dim = 128
latent_dim = 64

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Setup directories
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs('./plots', exist_ok=True)
if not os.path.exists(loss_log_file):
    with open(loss_log_file, 'w') as f:
        f.write('Epoch,D_Loss,G_Loss,Feature_Loss,Total_Loss,Real_Mean,Fake_Mean,Real_Std,Fake_Std\n')

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv('./flux_burnup_dataset.csv')
X = data[['Enrichment (%)', 'Flux', 'Burnup']].values

# Use RobustScaler for better handling of outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Save dataset stats for inference
data_center = scaler.center_
data_scale = scaler.scale_

print(f"Data shape: {X.shape}")
print(f"Data ranges - Original:")
print(f"  Enrichment: {X[:, 0].min():.2f} - {X[:, 0].max():.2f}")
print(f"  Flux: {X[:, 1].min():.2f} - {X[:, 1].max():.2f}")
print(f"  Burnup: {X[:, 2].min():.2e} - {X[:, 2].max():.2e}")

# Improved Generator with residual connections and attention
class ImprovedGenerator(nn.Module):
    def __init__(self, noise_dim=128, latent_dim=64):
        super().__init__()
        
        # Input processing
        self.noise_projection = nn.Linear(noise_dim, latent_dim)
        self.condition_projection = nn.Linear(3, latent_dim)
        
        # Main network with residual connections
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim * 2, 512)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        # Output layers for each feature
        self.enrichment_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Enrichment is percentage (0-100)
        )
        
        self.flux_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Flux is positive
        )
        
        self.burnup_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Burnup is positive
        )
        
        self.apply(self.init_weights)
    
    def forward(self, z, conditions):
        # Process noise and conditions
        z_proj = self.noise_projection(z)
        cond_proj = self.condition_projection(conditions)
        
        # Concatenate and process
        x = torch.cat([z_proj, cond_proj], dim=1)
        
        # Forward pass with residual connections
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        # Generate each feature separately
        enrichment = self.enrichment_head(x3) * 100  # Scale to 0-100
        flux = self.flux_head(x3) * 10  # Scale to reasonable flux range
        burnup = self.burnup_head(x3) * 1e-7  # Scale to burnup range
        
        return torch.cat([enrichment, flux, burnup], dim=1)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Improved Discriminator with feature matching
class ImprovedDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            spectral_norm(nn.Linear(3, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(128, 64)),
            nn.LeakyReLU(0.2)
        )
        
        # Discriminator head
        self.discriminator_head = nn.Sequential(
            spectral_norm(nn.Linear(64, 32)),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        
        # Feature matching head
        self.feature_head = nn.Sequential(
            spectral_norm(nn.Linear(64, 32)),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64)  # Output features for matching
        )
        
        self.apply(self.init_weights)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        validity = self.discriminator_head(features)
        feature_output = self.feature_head(features)
        return validity, feature_output
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Initialize models
generator = ImprovedGenerator(noise_dim, latent_dim).to(device)
discriminator = ImprovedDiscriminator().to(device)

# Optimizers with better learning rates
optimizer_G = optim.AdamW(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)

# Learning rate schedulers
scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=1e-6)
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=1e-6)

# Loss functions
adversarial_loss = nn.BCEWithLogitsLoss()
feature_loss = nn.MSELoss()

# Mixed precision scaler
scaler = GradScaler()

# Checkpoint functions
def save_checkpoint(epoch):
    path = os.path.join(checkpoint_dir, f'improved_checkpoint_{epoch}.tar')
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'data_center': data_center,
        'data_scale': data_scale
    }, path)
    print(f"[Checkpoint] Saved at epoch {epoch}")

def load_checkpoint():
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('improved_checkpoint_') and f.endswith('.tar')]
    if not files:
        print("[Checkpoint] No checkpoint found. Starting fresh.")
        return 0
    
    latest = max(files, key=lambda f: int(f.split('_')[2].split('.')[0]))
    path = os.path.join(checkpoint_dir, latest)
    
    try:
        checkpoint = torch.load(path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        print(f"[Checkpoint] Loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'] + 1
    except Exception as e:
        print(f"[Checkpoint] Error loading: {str(e)}. Starting fresh.")
        return 0

# Load checkpoint if exists
start_epoch = load_checkpoint()

# Convert data to tensors
dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Training loop with improved stability
print("Starting training...")
for epoch in range(start_epoch, num_epochs):
    epoch_d_losses = []
    epoch_g_losses = []
    epoch_feature_losses = []
    
    for batch_idx, real_data in enumerate(dataloader):
        real_data = real_data[0].to(device, non_blocking=True)
        current_batch_size = real_data.size(0)
        
        # Generate noise
        z = torch.randn(current_batch_size, noise_dim, device=device)
        
        # Train Discriminator
        optimizer_D.zero_grad(set_to_none=True)
        
        with autocast():
            # Real data
            real_validity, real_features = discriminator(real_data)
            real_labels = torch.full((current_batch_size, 1), 0.9, device=device)
            d_real_loss = adversarial_loss(real_validity, real_labels)
            
            # Fake data
            fake_data = generator(z, real_data)
            fake_validity, fake_features = discriminator(fake_data.detach())
            fake_labels = torch.zeros((current_batch_size, 1), device=device)
            d_fake_loss = adversarial_loss(fake_validity, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
        
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()
        
        # Train Generator
        if batch_idx % 2 == 0:  # Train generator every other batch
            optimizer_G.zero_grad(set_to_none=True)
            
            with autocast():
                # Generate fake data
                fake_data = generator(z, real_data)
                fake_validity, fake_features = discriminator(fake_data)
                
                # Adversarial loss
                gen_labels = torch.ones(current_batch_size, 1, device=device)
                g_loss = adversarial_loss(fake_validity, gen_labels)
                
                # Feature matching loss for better training stability
                feature_loss_val = feature_loss(fake_features, real_features.detach())
                
                # Total generator loss
                total_g_loss = g_loss + 0.1 * feature_loss_val
            
            scaler.scale(total_g_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()
            
            epoch_g_losses.append(total_g_loss.item())
            epoch_feature_losses.append(feature_loss_val.item())
        
        epoch_d_losses.append(d_loss.item())
    
    # Update learning rates
    scheduler_G.step()
    scheduler_D.step()
    
    # Calculate epoch statistics
    avg_d_loss = np.mean(epoch_d_losses)
    avg_g_loss = np.mean(epoch_g_losses) if epoch_g_losses else 0
    avg_feature_loss = np.mean(epoch_feature_losses) if epoch_feature_losses else 0
    
    # Generate sample statistics
    with torch.no_grad():
        z_sample = torch.randn(batch_size, noise_dim, device=device)
        sample_conditions = torch.randn(batch_size, 3, device=device)
        fake_samples = generator(z_sample, sample_conditions)
        
        real_mean = real_data.mean().item()
        fake_mean = fake_samples.mean().item()
        real_std = real_data.std().item()
        fake_std = fake_samples.std().item()
    
    # Logging
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | D: {avg_d_loss:.4f} | G: {avg_g_loss:.4f} | F: {avg_feature_loss:.4f} | "
              f"Real_μ: {real_mean:.3f} | Fake_μ: {fake_mean:.3f} | Real_σ: {real_std:.3f} | Fake_σ: {fake_std:.3f}")
    
    # Save to CSV
    with open(loss_log_file, 'a') as f:
        f.write(f'{epoch},{avg_d_loss},{avg_g_loss},{avg_feature_loss},{avg_g_loss + avg_feature_loss},'
                f'{real_mean},{fake_mean},{real_std},{fake_std}\n')
    
    # Checkpointing
    if epoch % checkpoint_interval == 0 and epoch > 0:
        save_checkpoint(epoch)

# Final save
save_checkpoint(num_epochs - 1)
print("Training completed!")

# Generate and save sample results
print("Generating sample results...")
generator.eval()
with torch.no_grad():
    z_test = torch.randn(1000, noise_dim, device=device)
    conditions_test = torch.randn(1000, 3, device=device)
    fake_samples = generator(z_test, conditions_test).cpu().numpy()
    
    # Convert back to original scale
    fake_samples_original = scaler.inverse_transform(fake_samples)
    
    # Save results
    results_df = pd.DataFrame(fake_samples_original, columns=['Enrichment (%)', 'Flux', 'Burnup'])
    results_df.to_csv('./plots/generated_samples.csv', index=False)
    
    # Print statistics
    print("\nGenerated Sample Statistics:")
    print(f"Enrichment: {fake_samples_original[:, 0].mean():.2f} ± {fake_samples_original[:, 0].std():.2f}")
    print(f"Flux: {fake_samples_original[:, 1].mean():.2f} ± {fake_samples_original[:, 1].std():.2f}")
    print(f"Burnup: {fake_samples_original[:, 2].mean():.2e} ± {fake_samples_original[:, 2].std():.2e}")
    
    print("\nOriginal Data Statistics:")
    print(f"Enrichment: {X[:, 0].mean():.2f} ± {X[:, 0].std():.2f}")
    print(f"Flux: {X[:, 1].mean():.2f} ± {X[:, 1].std():.2f}")
    print(f"Burnup: {X[:, 2].mean():.2e} ± {X[:, 2].std():.2e}")

print("Results saved to ./plots/generated_samples.csv") 