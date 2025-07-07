import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr, spearmanr
import os

def load_trained_model(checkpoint_path):
    """Load the trained generator model"""
    from improved_fluxgan import ImprovedGenerator, device
    
    generator = ImprovedGenerator(noise_dim=128, latent_dim=64).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    return generator, checkpoint['data_center'], checkpoint['data_scale']

def evaluate_model_accuracy(generator, data_center, data_scale, original_data_path):
    """Evaluate the accuracy of the generated samples"""
    
    # Load original data
    original_data = pd.read_csv(original_data_path)
    X_original = original_data[['Enrichment (%)', 'Flux', 'Burnup']].values
    
    # Generate samples
    generator.eval()
    with torch.no_grad():
        z = torch.randn(10000, 128, device=generator.device)
        conditions = torch.randn(10000, 3, device=generator.device)
        fake_samples = generator(z, conditions).cpu().numpy()
    
    # Convert back to original scale
    scaler = RobustScaler()
    scaler.center_ = data_center
    scaler.scale_ = data_scale
    fake_samples_original = scaler.inverse_transform(fake_samples)
    
    # Calculate accuracy metrics
    metrics = {}
    features = ['Enrichment (%)', 'Flux', 'Burnup']
    
    for i, feature in enumerate(features):
        real_values = X_original[:, i]
        fake_values = fake_samples_original[:, i]
        
        # Basic statistics
        metrics[f'{feature}_real_mean'] = float(real_values.mean())
        metrics[f'{feature}_fake_mean'] = float(fake_values.mean())
        metrics[f'{feature}_real_std'] = float(real_values.std())
        metrics[f'{feature}_fake_std'] = float(fake_values.std())
        
        # Distribution similarity (KL divergence approximation)
        real_hist, _ = np.histogram(real_values, bins=50, density=True)
        fake_hist, _ = np.histogram(fake_values, bins=50, density=True)
        
        # Avoid division by zero
        real_hist = np.maximum(real_hist, 1e-10)
        fake_hist = np.maximum(fake_hist, 1e-10)
        
        kl_div = np.sum(real_hist * np.log(real_hist / fake_hist))
        metrics[f'{feature}_kl_divergence'] = float(kl_div)
        
        # Correlation analysis
        if len(real_values) == len(fake_values):
            pearson_corr, _ = pearsonr(real_values, fake_values)
            spearman_corr, _ = spearmanr(real_values, fake_values)
            metrics[f'{feature}_pearson_corr'] = float(pearson_corr)
            metrics[f'{feature}_spearman_corr'] = float(spearman_corr)
    
    return metrics, fake_samples_original

def create_visualization_comparison(original_data, generated_data, save_path='./plots'):
    """Create comprehensive visualizations comparing original and generated data"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Original vs Generated Data Comparison', fontsize=16)
    
    features = ['Enrichment (%)', 'Flux', 'Burnup']
    
    for i, feature in enumerate(features):
        # Histograms
        axes[0, i].hist(original_data[:, i], bins=50, alpha=0.7, label='Original', color='blue', density=True)
        axes[0, i].hist(generated_data[:, i], bins=50, alpha=0.7, label='Generated', color='red', density=True)
        axes[0, i].set_title(f'{feature} Distribution')
        axes[0, i].set_xlabel(feature)
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Scatter plots
        if i == 0:  # Enrichment vs Flux
            axes[1, i].scatter(original_data[:, 0], original_data[:, 1], alpha=0.6, s=10, label='Original', color='blue')
            axes[1, i].scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.6, s=10, label='Generated', color='red')
            axes[1, i].set_xlabel('Enrichment (%)')
            axes[1, i].set_ylabel('Flux')
            axes[1, i].set_title('Enrichment vs Flux')
        elif i == 1:  # Enrichment vs Burnup
            axes[1, i].scatter(original_data[:, 0], original_data[:, 2], alpha=0.6, s=10, label='Original', color='blue')
            axes[1, i].scatter(generated_data[:, 0], generated_data[:, 2], alpha=0.6, s=10, label='Generated', color='red')
            axes[1, i].set_xlabel('Enrichment (%)')
            axes[1, i].set_ylabel('Burnup')
            axes[1, i].set_title('Enrichment vs Burnup')
        else:  # Flux vs Burnup
            axes[1, i].scatter(original_data[:, 1], original_data[:, 2], alpha=0.6, s=10, label='Original', color='blue')
            axes[1, i].scatter(generated_data[:, 1], generated_data[:, 2], alpha=0.6, s=10, label='Generated', color='red')
            axes[1, i].set_xlabel('Flux')
            axes[1, i].set_ylabel('Burnup')
            axes[1, i].set_title('Flux vs Burnup')
        
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'data_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create correlation heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data correlation
    original_df = pd.DataFrame(original_data, columns=features)
    corr_original = original_df.corr()
    sns.heatmap(corr_original, annot=True, cmap='coolwarm', center=0, ax=ax1, square=True)
    ax1.set_title('Original Data Correlation Matrix')
    
    # Generated data correlation
    generated_df = pd.DataFrame(generated_data, columns=features)
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
    
    features = ['Enrichment (%)', 'Flux', 'Burnup']
    
    for feature in features:
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
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT:")
    
    # Calculate average correlations
    avg_pearson = np.mean([metrics[f'{f}_pearson_corr'] for f in features])
    avg_spearman = np.mean([metrics[f'{f}_spearman_corr'] for f in features])
    avg_kl = np.mean([metrics[f'{f}_kl_divergence'] for f in features])
    
    print(f"Average Pearson Correlation: {avg_pearson:.4f}")
    print(f"Average Spearman Correlation: {avg_spearman:.4f}")
    print(f"Average KL Divergence: {avg_kl:.4f}")
    
    # Quality assessment
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
    X_original = original_data[['Enrichment (%)', 'Flux', 'Burnup']].values
    
    # Create visualizations
    create_visualization_comparison(X_original, generated_data)
    
    # Print accuracy report
    print_accuracy_report(metrics)
    
    # Save detailed results
    results_df = pd.DataFrame(generated_data, columns=['Enrichment (%)', 'Flux', 'Burnup'])
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