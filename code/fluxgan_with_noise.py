import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR

# Configuration
checkpoint_dir = './plots/checkpoint'
loss_log_file = './plots/loss_log.csv'
checkpoint_interval = 100
num_epochs = 10001
batch_size = 512
noise_dim = 100
label_flip_rate = 0.05   # 5% label flipping

# Setup CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Setup directories
os.makedirs(checkpoint_dir, exist_ok=True)
if not os.path.exists(loss_log_file):
    with open(loss_log_file, 'w') as f:
        f.write('Epoch,D Loss,G Loss,GenMean,GenStd\n')

# Load and preprocess data
# (Assumes flux_burnup_dataset.csv is in the same directory as this script)
data = pd.read_csv('./code/flux_burnup_dataset.csv')
X = data[['Enrichment (%)', 'Flux', 'Burnup']].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save dataset stats for inference
data_min = scaler.data_min_
data_max = scaler.data_max_

# Models for Vanilla GAN
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(noise_dim, 256)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.utils.spectral_norm(nn.Linear(128, 3)),
            nn.Tanh()
        )
        self.apply(self.init_weights)

    def forward(self, z):
        return self.net(z)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(3, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.LayerNorm(128),
            nn.utils.spectral_norm(nn.Linear(128, 1))
        )
        self.apply(self.init_weights)

    def forward(self, x):
        return self.net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Initialize models and move to GPU
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers with weight decay and adjusted learning rates
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999), weight_decay=1e-5)

# Learning Rate Scheduler (gentler decay)
scheduler_G = StepLR(optimizer_G, step_size=1000, gamma=0.8)
scheduler_D = StepLR(optimizer_D, step_size=1000, gamma=0.8)

# Loss function
adversarial_loss = nn.BCEWithLogitsLoss()

# Mixed Precision Scaler
scaler = GradScaler()

# Add instance noise function
def add_instance_noise(data, std=0.01):
    """Add Gaussian noise to data for instance noise regularization."""
    return data + torch.randn_like(data) * std

# Checkpoint functions
def save_checkpoint(epoch):
    path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar')
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'data_min': data_min,
        'data_max': data_max
    }, path)
    print(f"[Checkpoint] Saved at epoch {epoch}")

def load_checkpoint():
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.tar')]
    if not files:
        print("[Checkpoint] No checkpoint found. Starting fresh.")
        return 0
    latest = max(files, key=lambda f: int(f.split('_')[1].split('.')[0]))
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

# Convert data to CUDA tensors
dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Training loop
for epoch in range(start_epoch, num_epochs):
    for real_data in dataloader:
        real_data = real_data[0].to(device, non_blocking=True)
        current_batch_size = real_data.size(0)

        # ====== Train Discriminator ======
        optimizer_D.zero_grad(set_to_none=True)
        with autocast():
            # --- Label smoothing and label noise ---
            real_labels = torch.full((current_batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros((current_batch_size, 1), device=device)
            # Add label noise
            real_labels += 0.05 * torch.rand_like(real_labels)  # noise in [0, 0.05]
            fake_labels += 0.05 * torch.rand_like(fake_labels)  # noise in [0, 0.05]
            n_flip = int(label_flip_rate * current_batch_size)
            if n_flip > 0:
                idx_flip = torch.randperm(current_batch_size)[:n_flip]
                real_labels[idx_flip] = 0
                fake_labels[idx_flip] = 1

            # --- Instance noise ---
            real_data_noisy = add_instance_noise(real_data, std=0.01)

            real_output = discriminator(real_data_noisy)
            d_loss_real = adversarial_loss(real_output, real_labels)

            z = torch.randn(current_batch_size, noise_dim, device=device)
            fake_data = generator(z)
            fake_data_noisy = add_instance_noise(fake_data.detach(), std=0.01)
            fake_output = discriminator(fake_data_noisy)
            d_loss_fake = adversarial_loss(fake_output, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # ====== Train Generator (1:1 ratio) ======
        optimizer_G.zero_grad(set_to_none=True)
        with autocast():
            gen_labels = torch.ones(current_batch_size, 1, device=device)
            g_output = discriminator(fake_data)
            g_loss = adversarial_loss(g_output, gen_labels)
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

    # Step the learning rate schedulers
    scheduler_G.step()
    scheduler_D.step()

    # Logging and checkpointing
    if epoch % 1 == 0:
        with torch.no_grad():
            z_log = torch.randn(batch_size, noise_dim, device=device)
            gen_samples = generator(z_log).cpu().numpy()
            gen_mean, gen_std = gen_samples.mean(), gen_samples.std()
        print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | GenMean: {gen_mean:.3f} | GenStd: {gen_std:.3f}")
        with open(loss_log_file, 'a') as f:
            f.write(f'{epoch},{d_loss.item()},{g_loss.item()},{gen_mean},{gen_std}\n')
    if epoch % checkpoint_interval == 0 and epoch > 0:
        save_checkpoint(epoch)

# Final save
save_checkpoint(num_epochs - 1) 

enrichment_values = np.array([
    11.49, 3.5, 67.72, 17.36, 11.02, 69.62, 1.73, 42.41, 36.69, 18.87, 
    67.97, 63.78, 55.78, 20.99, 67.82, 60.98, 12.33, 56.87, 62.92, 5.0,
    10.35, 2.11, 64.83, 4.93, 26.02, 76.18, 61.37, 62.78, 35.71, 35.29,
    74.78, 48.16, 36.33, 62.18, 41.57, 32.45, 44.28, 29.37, 59.88, 31.42,
    61.48, 43.76, 60.84, 29.65, 50.77, 14.72, 56.37, 18.69, 59.08, 52.48,
    69.12, 2.92, 42.22, 62.93, 75.02, 14.64, 34.94, 9.09, 67.41, 49.67,
    58.28, 49.66, 45.65, 30.15, 71.5, 52.89, 15.55, 29.94, 47.47, 78.25,
    25.37, 3.81, 34.87, 20.77, 40.62, 63.6, 35.78, 47.05, 23.73, 51.87,
    29.0, 63.33, 66.46, 20.4, 14.71, 11.55, 57.87, 55.93, 7.57, 61.82,
    62.88, 70.7, 36.32, 75.0, 39.93, 54.81, 74.23, 24.26, 23.35, 79.74
])