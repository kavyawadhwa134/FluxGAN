import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# Configuration
checkpoint_dir = './plots/checkpoint'
output_csv = './generated_for_enrichment_cgan.csv'
noise_dim = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The 100 enrichment values from the interference notebook
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

# Generator definition (must match cGAN training)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, cond_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(noise_dim + cond_dim, 256)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.utils.spectral_norm(nn.Linear(128, 3)),
            nn.Tanh()
        )
        self.apply(self.init_weights)

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def load_latest_checkpoint():
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_cgan_') and f.endswith('.tar')]
    if not files:
        print('[Checkpoint] No cGAN checkpoint found. Cannot generate samples.')
        return None
    latest = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    path = os.path.join(checkpoint_dir, latest)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint

def normalize(value, data_min, data_max):
    return (value - data_min) / (data_max - data_min)

def denormalize(value, data_min, data_max):
    return value * (data_max - data_min) + data_min

if __name__ == '__main__':
    checkpoint = load_latest_checkpoint()
    if checkpoint is None:
        exit(1)

    # Load generator
    generator = Generator(noise_dim, cond_dim=1).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    # Load scaler min/max for denormalization
    data_min = checkpoint['data_min']
    data_max = checkpoint['data_max']

    results = []
    with torch.no_grad():
        for enrich in enrichment_values:
            # Normalize enrichment
            enrich_norm = normalize(enrich, data_min[0], data_max[0])
            enrich_tensor = torch.tensor([[enrich_norm]], dtype=torch.float32, device=device)
            z = torch.randn(1, noise_dim, device=device)
            gen_sample = generator(z, enrich_tensor).cpu().numpy()[0]
            # Denormalize
            enrich_out = denormalize(gen_sample[0], data_min[0], data_max[0])
            flux_out = denormalize(gen_sample[1], data_min[1], data_max[1])
            burnup_out = denormalize(gen_sample[2], data_min[2], data_max[2])
            results.append([enrich, flux_out, burnup_out])

    df = pd.DataFrame(results, columns=pd.Index(['Enrichment (%)', 'Flux', 'Burnup']))
    print(df)
    df.to_csv(output_csv, index=False)
    print(f'Generated samples for enrichment values saved to {output_csv}') 