import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import StepLR

# Configuration
checkpoint_dir = './plots/checkpoint_working'
loss_log_file = './plots/loss_log_working.csv'
checkpoint_interval = 1000
num_epochs = 15001
batch_size = 512
noise_dim = 100
label_flip_rate = 0.05

# Focus on working physics constraints
physics_weight = 0.02  # Small but not minimal
temperature_tolerance = 50.0  # Reasonable
burnup_correlation_weight = 0.01  # Small
thermal_hydraulics_weight = 0.01  # Small
fuel_performance_weight = 0.005  # Very small
# Removed neutronics_weight - this was causing issues

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Setup directories
os.makedirs(checkpoint_dir, exist_ok=True)
if not os.path.exists(loss_log_file):
    with open(loss_log_file, 'w') as f:
        f.write('Epoch,D Loss,G Loss,Temp Loss,Thermal Loss,Fuel Loss,Burnup Loss,GenMean,GenStd\n')

# Load and preprocess data
data = pd.read_csv('./code/flux_burnup_dataset.csv')
feature_cols = ['Enrichment (%)', 'Flux (n/cm²/s)', 'Burnup (MWd/kgU)', 
                'Fuel Centerline Temp (K)', 'Clad Surface Temp (K)', 'Coolant Outlet Temp (K)',
                'Reactivity', 'HTC (W/m2K)', 'FlowRate (kg/s)', 'Swelling (%)', 'FissionGasRelease (%)']
X = data[feature_cols].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save dataset stats for inference
data_min = scaler.data_min_
data_max = scaler.data_max_

class WorkingPhysicsConstraints:
    """Physics constraints that focus on working components"""
    
    def __init__(self, data_min, data_max):
        self.data_min = torch.tensor(data_min, device=device)
        self.data_max = torch.tensor(data_max, device=device)
        
    def denormalize(self, x_scaled):
        """Convert scaled data back to original scale"""
        return x_scaled * (self.data_max - self.data_min) + self.data_min
    
    def normalize(self, x_original):
        """Convert original scale data to scaled"""
        return (x_original - self.data_min) / (self.data_max - self.data_min)
    
    def temperature_consistency_loss(self, generated_data):
        """Temperature consistency with reasonable constraints"""
        denorm_data = self.denormalize(generated_data)
        
        fuel_temp = denorm_data[:, 3]
        clad_temp = denorm_data[:, 4]
        coolant_temp = denorm_data[:, 5]
        
        # Temperature ordering
        temp_order_loss = torch.mean(torch.relu(torch.clamp(clad_temp - fuel_temp + temperature_tolerance, max=100.0))) + \
                         torch.mean(torch.relu(torch.clamp(coolant_temp - clad_temp + temperature_tolerance, max=100.0)))
        
        # Temperature differences
        fuel_clad_diff = torch.abs(fuel_temp - clad_temp)
        clad_coolant_diff = torch.abs(clad_temp - coolant_temp)
        
        reasonable_diff_loss = torch.mean(torch.relu(torch.clamp(fuel_clad_diff - 20.0, max=50.0))) + \
                              torch.mean(torch.relu(torch.clamp(clad_coolant_diff - 80.0, max=100.0)))
        
        return temp_order_loss + reasonable_diff_loss
    
    def thermal_hydraulics_loss(self, generated_data):
        """Thermal-hydraulics constraints"""
        denorm_data = self.denormalize(generated_data)
        
        htc = denorm_data[:, 7]
        flow_rate = denorm_data[:, 8]
        
        # Reasonable bounds
        htc_bounds_loss = torch.mean(torch.relu(torch.clamp(500 - htc, max=50000))) + \
                         torch.mean(torch.relu(torch.clamp(htc - 100000, max=50000)))
        
        flow_bounds_loss = torch.mean(torch.relu(torch.clamp(2 - flow_rate, max=50))) + \
                          torch.mean(torch.relu(torch.clamp(flow_rate - 100, max=50)))
        
        # Correlation between HTC and flow rate
        try:
            htc_flow_corr = torch.corrcoef(torch.stack([htc, flow_rate]))[0, 1]
            if torch.isnan(htc_flow_corr):
                htc_flow_corr = torch.tensor(0.0, device=device)
            htc_correlation_loss = torch.relu(-htc_flow_corr)
        except:
            htc_correlation_loss = torch.tensor(0.0, device=device)
        
        return htc_bounds_loss + flow_bounds_loss + htc_correlation_loss
    
    def fuel_performance_loss(self, generated_data):
        """Fuel performance constraints"""
        denorm_data = self.denormalize(generated_data)
        
        burnup = denorm_data[:, 2]
        fuel_temp = denorm_data[:, 3]
        swelling = denorm_data[:, 9]
        fgr = denorm_data[:, 10]
        
        # Bounds
        swelling_bounds_loss = torch.mean(torch.relu(torch.clamp(-swelling, max=10.0))) + \
                              torch.mean(torch.relu(torch.clamp(swelling - 15.0, max=10.0)))
        
        fgr_bounds_loss = torch.mean(torch.relu(torch.clamp(-fgr, max=10.0))) + \
                         torch.mean(torch.relu(torch.clamp(fgr - 30.0, max=10.0)))
        
        # Correlations
        try:
            swelling_burnup_corr = torch.corrcoef(torch.stack([burnup, swelling]))[0, 1]
            if torch.isnan(swelling_burnup_corr):
                swelling_burnup_corr = torch.tensor(0.0, device=device)
            swelling_temp_corr = torch.corrcoef(torch.stack([fuel_temp, swelling]))[0, 1]
            if torch.isnan(swelling_temp_corr):
                swelling_temp_corr = torch.tensor(0.0, device=device)
            swelling_correlation_loss = torch.relu(-swelling_burnup_corr) + torch.relu(-swelling_temp_corr)
        except:
            swelling_correlation_loss = torch.tensor(0.0, device=device)
        
        try:
            fgr_burnup_corr = torch.corrcoef(torch.stack([burnup, fgr]))[0, 1]
            if torch.isnan(fgr_burnup_corr):
                fgr_burnup_corr = torch.tensor(0.0, device=device)
            fgr_temp_corr = torch.corrcoef(torch.stack([fuel_temp, fgr]))[0, 1]
            if torch.isnan(fgr_temp_corr):
                fgr_temp_corr = torch.tensor(0.0, device=device)
            fgr_correlation_loss = torch.relu(-fgr_burnup_corr) + torch.relu(-fgr_temp_corr)
        except:
            fgr_correlation_loss = torch.tensor(0.0, device=device)
        
        return swelling_bounds_loss + fgr_bounds_loss + swelling_correlation_loss + fgr_correlation_loss
    
    def burnup_flux_correlation_loss(self, generated_data):
        """Burnup-flux correlation"""
        denorm_data = self.denormalize(generated_data)
        
        burnup = denorm_data[:, 2]
        flux = denorm_data[:, 1]
        
        # Simple correlation check
        burnup_mean = torch.mean(burnup)
        flux_mean = torch.mean(flux)
        
        numerator = torch.mean((burnup - burnup_mean) * (flux - flux_mean))
        denominator = torch.std(burnup) * torch.std(flux) + 1e-8
        
        correlation = numerator / denominator
        
        if torch.isnan(correlation):
            correlation = torch.tensor(0.0, device=device)
        
        correlation_loss = torch.relu(-correlation)
        
        return correlation_loss
    
    def enrichment_effects_loss(self, generated_data, enrichment_cond):
        """Enrichment effects"""
        denorm_data = self.denormalize(generated_data)
        denorm_enrichment = self.denormalize(enrichment_cond)[:, 0]
        
        fuel_temp = denorm_data[:, 3]
        clad_temp = denorm_data[:, 4]
        
        # Simple correlation checks
        enrich_temp_corr = torch.corrcoef(torch.stack([denorm_enrichment, fuel_temp]))[0, 1]
        if torch.isnan(enrich_temp_corr):
            enrich_temp_corr = torch.tensor(0.0, device=device)
        enrich_clad_corr = torch.corrcoef(torch.stack([denorm_enrichment, clad_temp]))[0, 1]
        if torch.isnan(enrich_clad_corr):
            enrich_clad_corr = torch.tensor(0.0, device=device)
        
        correlation_loss = torch.relu(-enrich_temp_corr) + torch.relu(-enrich_clad_corr)
        
        return correlation_loss
    
    def physical_bounds_loss(self, generated_data):
        """Physical bounds"""
        denorm_data = self.denormalize(generated_data)
        
        bounds_loss = 0.0
        
        # Enrichment: 0.5-95%
        enrichment = denorm_data[:, 0]
        bounds_loss += torch.mean(torch.relu(torch.clamp(0.5 - enrichment, max=50.0))) + \
                      torch.mean(torch.relu(torch.clamp(enrichment - 95.0, max=50.0)))
        
        # Flux: 1e12 - 1e15 n/cm²/s
        flux = denorm_data[:, 1]
        bounds_loss += torch.mean(torch.relu(torch.clamp(1e12 - flux, max=1e15))) + \
                      torch.mean(torch.relu(torch.clamp(flux - 1e15, max=1e15)))
        
        # Burnup: 0-100 MWd/kgU
        burnup = denorm_data[:, 2]
        bounds_loss += torch.mean(torch.relu(torch.clamp(-burnup, max=50.0))) + \
                      torch.mean(torch.relu(torch.clamp(burnup - 100.0, max=50.0)))
        
        # Temperatures: 300-2500K
        for i in [3, 4, 5]:
            temp = denorm_data[:, i]
            bounds_loss += torch.mean(torch.relu(torch.clamp(300.0 - temp, max=500.0))) + \
                          torch.mean(torch.relu(torch.clamp(temp - 2500.0, max=500.0)))
        
        return bounds_loss

# Initialize physics constraints
physics_constraints = WorkingPhysicsConstraints(data_min, data_max)

# Generator with good initialization
class WorkingGenerator(nn.Module):
    def __init__(self, noise_dim=100, cond_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.Linear(128, 11),
            nn.Tanh()
        )
        self.apply(self.init_weights)

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.4)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Discriminator with good initialization
class WorkingDiscriminator(nn.Module):
    def __init__(self, cond_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11 + cond_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        self.apply(self.init_weights)

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.4)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Initialize models
generator = WorkingGenerator(noise_dim, cond_dim=1).to(device)
discriminator = WorkingDiscriminator(cond_dim=1).to(device)

# Optimizers with good learning rates
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-6)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999), weight_decay=1e-6)

# Learning Rate Scheduler
scheduler_G = StepLR(optimizer_G, step_size=1500, gamma=0.9)
scheduler_D = StepLR(optimizer_D, step_size=1500, gamma=0.9)

# Loss functions
adversarial_loss = nn.BCEWithLogitsLoss()

# Mixed Precision Scaler
scaler = GradScaler()

def add_instance_noise(data, std=0.003):  # Small noise
    """Add Gaussian noise to data for instance noise regularization."""
    return data + torch.randn_like(data) * std

# Checkpoint functions
def save_checkpoint(epoch):
    path = os.path.join(checkpoint_dir, f'checkpoint_working_{epoch}.tar')
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
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_working_') and f.endswith('.tar')]
    if not files:
        print("[Checkpoint] No checkpoint found. Starting fresh.")
        return 0
    latest = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    path = os.path.join(checkpoint_dir, latest)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        print(f"[Checkpoint] Loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'] + 1
    except Exception as e:
        print(f"[Checkpoint] Error loading: {str(e)}. Starting fresh.")
        return 0

# Prepare dataset
enrichment = X_scaled[:, 0:1]
targets = X_scaled
dataset = TensorDataset(torch.tensor(targets, dtype=torch.float32), torch.tensor(enrichment, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Load checkpoint if exists
start_epoch = load_checkpoint()

# Training loop
for epoch in range(start_epoch, num_epochs):
    for real_data, cond in dataloader:
        real_data = real_data.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        current_batch_size = real_data.size(0)

        # ====== Train Discriminator ======
        optimizer_D.zero_grad(set_to_none=True)
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # Label smoothing and noise
            real_labels = torch.full((current_batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros((current_batch_size, 1), device=device)
            real_labels += 0.05 * torch.rand_like(real_labels)
            fake_labels += 0.05 * torch.rand_like(fake_labels)
            
            n_flip = int(label_flip_rate * current_batch_size)
            if n_flip > 0:
                idx_flip = torch.randperm(current_batch_size)[:n_flip]
                real_labels[idx_flip] = 0
                fake_labels[idx_flip] = 1

            # Instance noise
            real_data_noisy = add_instance_noise(real_data, std=0.003)

            real_output = discriminator(real_data_noisy, cond)
            d_loss_real = adversarial_loss(real_output, real_labels)

            z = torch.randn(current_batch_size, noise_dim, device=device)
            fake_data = generator(z, cond)
            fake_data_noisy = add_instance_noise(fake_data.detach(), std=0.003)
            fake_output = discriminator(fake_data_noisy, cond)
            d_loss_fake = adversarial_loss(fake_output, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # ====== Train Generator with Working Physics Loss ======
        optimizer_G.zero_grad(set_to_none=True)
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            gen_labels = torch.ones(current_batch_size, 1, device=device)
            g_output = discriminator(fake_data, cond)
            g_loss_adversarial = adversarial_loss(g_output, gen_labels)
            
            # Working physics-informed losses (no neutronics)
            temp_loss = physics_constraints.temperature_consistency_loss(fake_data)
            thermal_loss = physics_constraints.thermal_hydraulics_loss(fake_data)
            fuel_loss = physics_constraints.fuel_performance_loss(fake_data)
            burnup_flux_loss = physics_constraints.burnup_flux_correlation_loss(fake_data)
            enrichment_loss = physics_constraints.enrichment_effects_loss(fake_data, cond)
            bounds_loss = physics_constraints.physical_bounds_loss(fake_data)
            
            # Combined physics loss without neutronics
            physics_loss = (temp_loss + 
                          thermal_hydraulics_weight * thermal_loss +
                          fuel_performance_weight * fuel_loss +
                          burnup_correlation_weight * burnup_flux_loss + 
                          enrichment_loss + 
                          bounds_loss)
            
            # Total generator loss
            g_loss = g_loss_adversarial + physics_weight * physics_loss

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
            cond_log = torch.rand(batch_size, 1, device=device)
            gen_samples = generator(z_log, cond_log).cpu().numpy()
            gen_mean, gen_std = gen_samples.mean(), gen_samples.std()
            
            # Calculate physics losses for logging
            fake_data_log = torch.tensor(gen_samples, device=device)
            temp_loss_log = physics_constraints.temperature_consistency_loss(fake_data_log).item()
            thermal_loss_log = physics_constraints.thermal_hydraulics_loss(fake_data_log).item()
            fuel_loss_log = physics_constraints.fuel_performance_loss(fake_data_log).item()
            burnup_loss_log = physics_constraints.burnup_flux_correlation_loss(fake_data_log).item()
            
        print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Temp: {temp_loss_log:.4f} | Thermal: {thermal_loss_log:.4f} | Fuel: {fuel_loss_log:.4f} | Burnup: {burnup_loss_log:.4f} | GenMean: {gen_mean:.3f} | GenStd: {gen_std:.3f}")
        with open(loss_log_file, 'a') as f:
            f.write(f'{epoch},{d_loss.item()},{g_loss.item()},{temp_loss_log},{thermal_loss_log},{fuel_loss_log},{burnup_loss_log},{gen_mean},{gen_std}\n')
    
    if epoch % checkpoint_interval == 0 and epoch > 0:
        save_checkpoint(epoch)

# Final save
save_checkpoint(num_epochs - 1)
print("Working FluxGAN training completed!") 