import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

# Load the trained model
checkpoint_dir = './plots/checkpoint_working'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator class (same as training)
class WorkingGenerator(torch.nn.Module):
    def __init__(self, noise_dim=100, cond_dim=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(noise_dim + cond_dim, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 11),
            torch.nn.Tanh()
        )

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.net(x)

def load_latest_checkpoint():
    """Load the latest checkpoint"""
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_working_') and f.endswith('.tar')]
    if not files:
        raise FileNotFoundError("No checkpoints found")
    
    latest = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    path = os.path.join(checkpoint_dir, latest)
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint

def generate_samples(generator, num_samples=10000, enrichment_range=(0.7, 0.9)):
    """Generate samples with the trained model"""
    generator.eval()
    
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(num_samples, 100, device=device)
        
        # Generate enrichment conditions
        enrichment_cond = torch.rand(num_samples, 1, device=device) * (enrichment_range[1] - enrichment_range[0]) + enrichment_range[0]
        
        # Generate samples
        samples = generator(z, enrichment_cond)
        
        return samples.cpu().numpy(), enrichment_cond.cpu().numpy()

def denormalize_samples(samples, data_min, data_max):
    """Convert normalized samples back to original scale"""
    return samples * (data_max - data_min) + data_min

def analyze_generated_samples(generated_data, real_data, feature_names):
    """Analyze the generated samples"""
    
    # Create DataFrames
    gen_df = pd.DataFrame(generated_data, columns=feature_names)
    real_df = pd.DataFrame(real_data, columns=feature_names)
    
    print("=" * 60)
    print("GENERATED SAMPLES ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print("\n📊 GENERATED SAMPLES STATISTICS:")
    print(gen_df.describe())
    
    print("\n📊 REAL DATA STATISTICS:")
    print(real_df.describe())
    
    # Check for physical bounds
    print("\n🔬 PHYSICAL BOUNDS CHECK:")
    
    # Enrichment bounds (0.5-95%)
    enrich_violations = ((gen_df['Enrichment (%)'] < 0.5) | (gen_df['Enrichment (%)'] > 95)).sum()
    print(f"Enrichment violations: {enrich_violations}/{len(gen_df)} ({enrich_violations/len(gen_df)*100:.2f}%)")
    
    # Flux bounds (1e12 - 1e15)
    flux_violations = ((gen_df['Flux (n/cm²/s)'] < 1e12) | (gen_df['Flux (n/cm²/s)'] > 1e15)).sum()
    print(f"Flux violations: {flux_violations}/{len(gen_df)} ({flux_violations/len(gen_df)*100:.2f}%)")
    
    # Temperature ordering
    temp_order_violations = ((gen_df['Clad Surface Temp (K)'] > gen_df['Fuel Centerline Temp (K)']) | 
                           (gen_df['Coolant Outlet Temp (K)'] > gen_df['Clad Surface Temp (K)'])).sum()
    print(f"Temperature ordering violations: {temp_order_violations}/{len(gen_df)} ({temp_order_violations/len(gen_df)*100:.2f}%)")
    
    # Temperature bounds (300-2500K)
    temp_violations = 0
    for temp_col in ['Fuel Centerline Temp (K)', 'Clad Surface Temp (K)', 'Coolant Outlet Temp (K)']:
        temp_violations += ((gen_df[temp_col] < 300) | (gen_df[temp_col] > 2500)).sum()
    print(f"Temperature bounds violations: {temp_violations}/{len(gen_df)*3} ({temp_violations/(len(gen_df)*3)*100:.2f}%)")
    
    return gen_df, real_df

def create_comparison_plots(gen_df, real_df, feature_names):
    """Create comparison plots between generated and real data"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots for each feature
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        
        # Plot histograms
        ax.hist(real_df[feature], bins=50, alpha=0.7, label='Real Data', density=True, color='blue')
        ax.hist(gen_df[feature], bins=50, alpha=0.7, label='Generated Data', density=True, color='red')
        
        ax.set_title(f'{feature} Distribution')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('./plots/generated_vs_real_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create correlation heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Real data correlation
    real_corr = real_df.corr()
    sns.heatmap(real_corr, annot=True, cmap='coolwarm', center=0, ax=ax1, fmt='.2f')
    ax1.set_title('Real Data Correlation Matrix')
    
    # Generated data correlation
    gen_corr = gen_df.corr()
    sns.heatmap(gen_corr, annot=True, cmap='coolwarm', center=0, ax=ax2, fmt='.2f')
    ax2.set_title('Generated Data Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('./plots/correlation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Temperature relationships
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Fuel vs Clad temperature
    axes[0].scatter(real_df['Fuel Centerline Temp (K)'], real_df['Clad Surface Temp (K)'], 
                   alpha=0.6, label='Real', s=20)
    axes[0].scatter(gen_df['Fuel Centerline Temp (K)'], gen_df['Clad Surface Temp (K)'], 
                   alpha=0.6, label='Generated', s=20)
    axes[0].plot([300, 2500], [300, 2500], 'k--', alpha=0.5)
    axes[0].set_xlabel('Fuel Centerline Temp (K)')
    axes[0].set_ylabel('Clad Surface Temp (K)')
    axes[0].set_title('Fuel vs Clad Temperature')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Clad vs Coolant temperature
    axes[1].scatter(real_df['Clad Surface Temp (K)'], real_df['Coolant Outlet Temp (K)'], 
                   alpha=0.6, label='Real', s=20)
    axes[1].scatter(gen_df['Clad Surface Temp (K)'], gen_df['Coolant Outlet Temp (K)'], 
                   alpha=0.6, label='Generated', s=20)
    axes[1].plot([300, 2500], [300, 2500], 'k--', alpha=0.5)
    axes[1].set_xlabel('Clad Surface Temp (K)')
    axes[1].set_ylabel('Coolant Outlet Temp (K)')
    axes[1].set_title('Clad vs Coolant Temperature')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Burnup vs Flux
    axes[2].scatter(real_df['Burnup (MWd/kgU)'], real_df['Flux (n/cm²/s)'], 
                   alpha=0.6, label='Real', s=20)
    axes[2].scatter(gen_df['Burnup (MWd/kgU)'], gen_df['Flux (n/cm²/s)'], 
                   alpha=0.6, label='Generated', s=20)
    axes[2].set_xlabel('Burnup (MWd/kgU)')
    axes[2].set_ylabel('Flux (n/cm²/s)')
    axes[2].set_title('Burnup vs Flux')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./plots/physics_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 Loading trained model and generating samples...")
    
    # Load checkpoint
    checkpoint = load_latest_checkpoint()
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Initialize generator
    generator = WorkingGenerator(noise_dim=100, cond_dim=1).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # Load original data for comparison
    data = pd.read_csv('./code/flux_burnup_dataset.csv')
    feature_cols = ['Enrichment (%)', 'Flux (n/cm²/s)', 'Burnup (MWd/kgU)', 
                    'Fuel Centerline Temp (K)', 'Clad Surface Temp (K)', 'Coolant Outlet Temp (K)',
                    'Reactivity', 'HTC (W/m2K)', 'FlowRate (kg/s)', 'Swelling (%)', 'FissionGasRelease (%)']
    
    real_data = data[feature_cols].values
    
    # Generate samples
    print("🎲 Generating 10,000 samples...")
    generated_samples, enrichment_conditions = generate_samples(generator, num_samples=10000)
    
    # Denormalize samples
    data_min = checkpoint['data_min']
    data_max = checkpoint['data_max']
    generated_data = denormalize_samples(generated_samples, data_min, data_max)
    
    # Analyze samples
    gen_df, real_df = analyze_generated_samples(generated_data, real_data, feature_cols)
    
    # Create plots
    print("\n📈 Creating comparison plots...")
    create_comparison_plots(gen_df, real_df, feature_cols)
    
    # Save generated samples
    gen_df.to_csv('./generated_samples_working.csv', index=False)
    print(f"\n💾 Generated samples saved to: ./generated_samples_working.csv")
    
    print("\n✅ Analysis complete! Check the plots in ./plots/ directory.")
    
    # Summary statistics
    print("\n📋 SUMMARY:")
    print(f"Generated samples: {len(gen_df)}")
    print(f"Real data samples: {len(real_df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Training epochs: {checkpoint['epoch']}")

if __name__ == "__main__":
    main() 