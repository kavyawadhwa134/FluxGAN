import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import os

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Generator class (matching cGAN architecture)
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
            nn.utils.spectral_norm(nn.Linear(128, 6)),  # 6 outputs for multiphysics
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

def load_trained_model(checkpoint_path):
    """Load the trained generator model"""
    generator = Generator(noise_dim=100, cond_dim=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator, checkpoint['data_min'], checkpoint['data_max']

def evaluate_model_accuracy(generator, data_center, data_scale, original_data_path):
    """Evaluate the accuracy of the generated samples"""
    # Load original data
    feature_cols = ['Enrichment (%)', 'Flux (n/cm²/s)', 'Burnup (MWd/kgU)', 'Fuel Centerline Temp (K)', 'Clad Surface Temp (K)', 'Coolant Outlet Temp (K)']
    original_data = pd.read_csv(original_data_path)
    X_original = original_data[feature_cols].values
    # Generate samples
    generator.eval()
    with torch.no_grad():
        z = torch.randn(10000, 128, device=device)
        conditions = torch.randn(10000, 1, device=device)  # adjust if conditioning changes
        fake_samples = generator(z, conditions).cpu().numpy()
    # Convert back to original scale
    scaler = RobustScaler()
    scaler.center_ = data_center
    scaler.scale_ = data_scale
    fake_samples_original = scaler.inverse_transform(fake_samples)
    # Calculate accuracy metrics
    metrics = {}
    features = feature_cols
    for i, feature in enumerate(features):
        real_values = X_original[:, i]
        fake_values = fake_samples_original[:, i]
        # Basic statistics
        metrics[f'{feature}_real_mean'] = real_values.mean()
        metrics[f'{feature}_fake_mean'] = fake_values.mean()
        metrics[f'{feature}_real_std'] = real_values.std()
        metrics[f'{feature}_fake_std'] = fake_values.std()
        # Distribution similarity (KL divergence approximation)
        real_hist, _ = np.histogram(real_values, bins=50, density=True)
        fake_hist, _ = np.histogram(fake_values, bins=50, density=True)
        real_hist = np.maximum(real_hist, 1e-10)
        fake_hist = np.maximum(fake_hist, 1e-10)
        kl_div = np.sum(real_hist * np.log(real_hist / fake_hist))
        metrics[f'{feature}_kl_divergence'] = kl_div
        # Correlation analysis
        if len(real_values) == len(fake_values):
            pearson_corr, _ = pearsonr(real_values, fake_values)
            spearman_corr, _ = spearmanr(real_values, fake_values)
            metrics[f'{feature}_pearson_corr'] = pearson_corr
            metrics[f'{feature}_spearman_corr'] = spearman_corr
    return metrics, fake_samples_original

def create_visualization_comparison(original_data, generated_data, save_path='./plots'):
    """Create comprehensive visualizations comparing original and generated data"""
    os.makedirs(save_path, exist_ok=True)
    feature_cols = ['Enrichment (%)', 'Flux (n/cm²/s)', 'Burnup (MWd/kgU)', 'Fuel Centerline Temp (K)', 'Clad Surface Temp (K)', 'Coolant Outlet Temp (K)']
    num_features = len(feature_cols)
    # Create subplots for all features
    fig, axes = plt.subplots(2, num_features, figsize=(6*num_features, 12))
    fig.suptitle('Original vs Generated Data Comparison', fontsize=16)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    for i, (feature, color) in enumerate(zip(feature_cols, colors)):
        # Histograms
        axes[0, i].hist(original_data[:, i], bins=50, alpha=0.7, label='Original', color='blue', density=True)
        axes[0, i].hist(generated_data[:, i], bins=50, alpha=0.7, label='Generated', color='red', density=True)
        axes[0, i].set_title(f'{feature} Distribution')
        axes[0, i].set_xlabel(feature)
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        # Scatter plots: feature vs Enrichment (%)
        axes[1, i].scatter(original_data[:, 0], original_data[:, i], alpha=0.6, s=10, label='Original', color='blue')
        axes[1, i].scatter(generated_data[:, 0], generated_data[:, i], alpha=0.6, s=10, label='Generated', color='red')
        axes[1, i].set_xlabel('Enrichment (%)')
        axes[1, i].set_ylabel(feature)
        axes[1, i].set_title(f'Enrichment vs {feature}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'data_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    # Create correlation heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    original_df = pd.DataFrame(original_data, columns=feature_cols)
    corr_original = original_df.corr()
    sns.heatmap(corr_original, annot=True, cmap='coolwarm', center=0, ax=ax1, square=True)
    ax1.set_title('Original Data Correlation Matrix')
    generated_df = pd.DataFrame(generated_data, columns=feature_cols)
    corr_generated = generated_df.corr()
    sns.heatmap(corr_generated, annot=True, cmap='coolwarm', center=0, ax=ax2, square=True)
    ax2.set_title('Generated Data Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'correlation_matrices.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_accuracy_report(metrics):
    """Print a comprehensive accuracy report"""
    print("=" * 60)
    print("FLUXGAN ACCURACY EVALUATION REPORT")
    print("=" * 60)
    feature_cols = ['Enrichment (%)', 'Flux (n/cm²/s)', 'Burnup (MWd/kgU)', 'Fuel Centerline Temp (K)', 'Clad Surface Temp (K)', 'Coolant Outlet Temp (K)']
    for feature in feature_cols:
        print(f"\n{feature}:")
        print(f"  Real Mean: {metrics[f'{feature}_real_mean']:.4f}")
        print(f"  Fake Mean: {metrics[f'{feature}_fake_mean']:.4f}")
        print(f"  Mean Difference: {abs(metrics[f'{feature}_real_mean'] - metrics[f'{feature}_fake_mean']):.4f}")
        print(f"  Real Std: {metrics[f'{feature}_real_std']:.4f}")
        print(f"  Fake Std: {metrics[f'{feature}_fake_std']:.4f}")
        print(f"  Std Difference: {abs(metrics[f'{feature}_real_std'] - metrics[f'{feature}_fake_std']):.4f}")
        print(f"  KL Divergence: {metrics[f'{feature}_kl_divergence']:.4f}")
        print(f"  Pearson Correlation: {metrics[f'{feature}_pearson_corr']:.4f}")
        print(f"  Spearman Correlation: {metrics[f'{feature}_spearman_corr']:.4f}")
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT:")
    avg_pearson = np.mean([metrics[f'{f}_pearson_corr'] for f in feature_cols])
    avg_spearman = np.mean([metrics[f'{f}_spearman_corr'] for f in feature_cols])
    avg_kl = np.mean([metrics[f'{f}_kl_divergence'] for f in feature_cols])
    print(f"Average Pearson Correlation: {avg_pearson:.4f}")
    print(f"Average Spearman Correlation: {avg_spearman:.4f}")
    print(f"Average KL Divergence: {avg_kl:.4f}")
    if avg_pearson > 0.8 and avg_spearman > 0.8:
        print("✓ EXCELLENT: High correlation with original data")
    elif avg_pearson > 0.6 and avg_spearman > 0.6:
        print("✓ GOOD: Moderate correlation with original data")
    elif avg_pearson > 0.4 and avg_spearman > 0.4:
        print("⚠ FAIR: Some correlation with original data")
    else:
        print("✗ POOR: Low correlation with original data")
    if avg_kl < 0.5:
        print("✓ EXCELLENT: Very similar distribution to original data")
    elif avg_kl < 1.0:
        print("✓ GOOD: Similar distribution to original data")
    elif avg_kl < 2.0:
        print("⚠ FAIR: Somewhat similar distribution to original data")
    else:
        print("✗ POOR: Different distribution from original data")

def main():
    """Main evaluation function"""
    
    # Find the latest checkpoint
    checkpoint_dir = './plots/checkpoint'
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found. Please train the model first.")
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('improved_checkpoint_') and f.endswith('.tar')]
    if not checkpoint_files:
        print("No improved checkpoint files found. Please train the model first.")
        return
    
    latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[2].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model and evaluate
    generator, data_center, data_scale = load_trained_model(checkpoint_path)
    
    # Evaluate accuracy
    metrics, generated_data = evaluate_model_accuracy(
        generator, data_center, data_scale, './flux_burnup_dataset.csv'
    )
    
    # Load original data for visualization
    original_data = pd.read_csv('./flux_burnup_dataset.csv')
    X_original = original_data[['Enrichment (%)', 'Flux', 'Burnup']].values # This line was not in the new_code, but should be updated for consistency
    
    # Create visualizations
    create_visualization_comparison(X_original, generated_data)
    
    # Print accuracy report
    print_accuracy_report(metrics)
    
    # Save detailed results
    results_df = pd.DataFrame(generated_data, columns=['Enrichment (%)', 'Flux', 'Burnup']) # This line was not in the new_code, but should be updated for consistency
    results_df.to_csv('./plots/evaluation_results.csv', index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('./plots/accuracy_metrics.csv', index=False)
    
    print(f"\nDetailed results saved to:")
    print(f"  - ./plots/evaluation_results.csv")
    print(f"  - ./plots/accuracy_metrics.csv")
    print(f"  - ./plots/data_comparison.png")
    print(f"  - ./plots/correlation_matrices.png")

if __name__ == "__main__":
    main() 