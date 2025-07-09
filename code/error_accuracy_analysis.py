import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr, ks_2samp
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

def calculate_distribution_metrics(real_data, gen_data, feature_names):
    """Calculate distribution similarity metrics"""
    metrics = {}
    
    for i, feature in enumerate(feature_names):
        real_feature = real_data[:, i]
        gen_feature = gen_data[:, i]
        
        # Basic statistics comparison
        real_mean, real_std = np.mean(real_feature), np.std(real_feature)
        gen_mean, gen_std = np.mean(gen_feature), np.std(gen_feature)
        
        # Mean and std errors
        mean_error = abs(real_mean - gen_mean) / abs(real_mean) * 100
        std_error = abs(real_std - gen_std) / abs(real_std) * 100
        
        # Kolmogorov-Smirnov test for distribution similarity
        ks_stat, ks_pvalue = ks_2samp(real_feature, gen_feature)
        
        # Wasserstein distance (Earth Mover's Distance)
        from scipy.stats import wasserstein_distance
        wasserstein_dist = wasserstein_distance(real_feature, gen_feature)
        
        metrics[feature] = {
            'mean_error_%': mean_error,
            'std_error_%': std_error,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'wasserstein_distance': wasserstein_dist,
            'real_mean': real_mean,
            'real_std': real_std,
            'gen_mean': gen_mean,
            'gen_std': gen_std
        }
    
    return metrics

def calculate_correlation_metrics(real_df, gen_df):
    """Calculate correlation matrix accuracy"""
    real_corr = real_df.corr()
    gen_corr = gen_df.corr()
    
    # Flatten correlation matrices
    real_corr_flat = real_corr.values.flatten()
    gen_corr_flat = gen_corr.values.flatten()
    
    # Remove diagonal elements (always 1.0)
    mask = ~np.eye(real_corr.shape[0], dtype=bool)
    real_corr_flat = real_corr.values[mask]
    gen_corr_flat = gen_corr.values[mask]
    
    # Calculate correlation accuracy metrics
    mae_corr = mean_absolute_error(real_corr_flat, gen_corr_flat)
    mse_corr = mean_squared_error(real_corr_flat, gen_corr_flat)
    r2_corr = r2_score(real_corr_flat, gen_corr_flat)
    pearson_corr, pearson_p = pearsonr(real_corr_flat, gen_corr_flat)
    
    return {
        'correlation_mae': mae_corr,
        'correlation_mse': mse_corr,
        'correlation_r2': r2_corr,
        'correlation_pearson': pearson_corr,
        'correlation_pearson_p': pearson_p
    }

def calculate_physics_accuracy(gen_df):
    """Calculate physics constraint accuracy"""
    physics_metrics = {}
    
    # Temperature ordering accuracy
    temp_order_correct = ((gen_df['Fuel Centerline Temp (K)'] >= gen_df['Clad Surface Temp (K)']) & 
                         (gen_df['Clad Surface Temp (K)'] >= gen_df['Coolant Outlet Temp (K)'])).sum()
    temp_order_accuracy = temp_order_correct / len(gen_df) * 100
    physics_metrics['temperature_ordering_accuracy_%'] = temp_order_accuracy
    
    # Temperature difference accuracy
    fuel_clad_diff = gen_df['Fuel Centerline Temp (K)'] - gen_df['Clad Surface Temp (K)']
    clad_coolant_diff = gen_df['Clad Surface Temp (K)'] - gen_df['Coolant Outlet Temp (K)']
    
    reasonable_fuel_clad = ((fuel_clad_diff >= 0) & (fuel_clad_diff <= 100)).sum()
    reasonable_clad_coolant = ((clad_coolant_diff >= 0) & (clad_coolant_diff <= 200)).sum()
    
    physics_metrics['fuel_clad_diff_accuracy_%'] = reasonable_fuel_clad / len(gen_df) * 100
    physics_metrics['clad_coolant_diff_accuracy_%'] = reasonable_clad_coolant / len(gen_df) * 100
    
    # Enrichment bounds accuracy
    enrich_bounds_correct = ((gen_df['Enrichment (%)'] >= 0.5) & (gen_df['Enrichment (%)'] <= 95)).sum()
    physics_metrics['enrichment_bounds_accuracy_%'] = enrich_bounds_correct / len(gen_df) * 100
    
    # Flux bounds accuracy (log scale)
    flux_bounds_correct = ((gen_df['Flux (n/cm²/s)'] >= 6.0) & (gen_df['Flux (n/cm²/s)'] <= 12.0)).sum()
    physics_metrics['flux_bounds_accuracy_%'] = flux_bounds_correct / len(gen_df) * 100
    
    # Temperature bounds accuracy
    temp_bounds_correct = 0
    for temp_col in ['Fuel Centerline Temp (K)', 'Clad Surface Temp (K)', 'Coolant Outlet Temp (K)']:
        temp_bounds_correct += ((gen_df[temp_col] >= 300) & (gen_df[temp_col] <= 2500)).sum()
    physics_metrics['temperature_bounds_accuracy_%'] = temp_bounds_correct / (len(gen_df) * 3) * 100
    
    # Burnup bounds accuracy
    burnup_bounds_correct = ((gen_df['Burnup (MWd/kgU)'] >= 0) & (gen_df['Burnup (MWd/kgU)'] <= 100)).sum()
    physics_metrics['burnup_bounds_accuracy_%'] = burnup_bounds_correct / len(gen_df) * 100
    
    return physics_metrics

def calculate_overall_accuracy(physics_metrics):
    """Calculate overall accuracy score"""
    # Weight different physics constraints
    weights = {
        'temperature_ordering_accuracy_%': 0.25,
        'fuel_clad_diff_accuracy_%': 0.15,
        'clad_coolant_diff_accuracy_%': 0.15,
        'enrichment_bounds_accuracy_%': 0.15,
        'flux_bounds_accuracy_%': 0.15,
        'temperature_bounds_accuracy_%': 0.10,
        'burnup_bounds_accuracy_%': 0.05
    }
    
    overall_score = 0
    for metric, weight in weights.items():
        if metric in physics_metrics:
            overall_score += physics_metrics[metric] * weight
    
    return overall_score

def create_error_plots(distribution_metrics, correlation_metrics, physics_metrics):
    """Create error and accuracy visualization plots"""
    
    # Distribution accuracy plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mean error by feature
    features = list(distribution_metrics.keys())
    mean_errors = [distribution_metrics[f]['mean_error_%'] for f in features]
    std_errors = [distribution_metrics[f]['std_error_%'] for f in features]
    
    x = np.arange(len(features))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, mean_errors, width, label='Mean Error (%)', alpha=0.8)
    axes[0, 0].bar(x + width/2, std_errors, width, label='Std Error (%)', alpha=0.8)
    axes[0, 0].set_xlabel('Features')
    axes[0, 0].set_ylabel('Error (%)')
    axes[0, 0].set_title('Distribution Accuracy by Feature')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(features, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # KS test statistics
    ks_stats = [distribution_metrics[f]['ks_statistic'] for f in features]
    axes[0, 1].bar(features, ks_stats, alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Features')
    axes[0, 1].set_ylabel('KS Statistic')
    axes[0, 1].set_title('Kolmogorov-Smirnov Test Statistics')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Physics accuracy
    physics_names = list(physics_metrics.keys())
    physics_values = list(physics_metrics.values())
    
    axes[1, 0].bar(physics_names, physics_values, alpha=0.8, color='green')
    axes[1, 0].set_xlabel('Physics Constraints')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Physics Constraint Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overall accuracy summary
    overall_score = calculate_overall_accuracy(physics_metrics)
    
    # Create a summary table
    summary_data = {
        'Metric': ['Overall Physics Accuracy', 'Correlation R²', 'Correlation MAE', 'Mean Distribution Error'],
        'Value': [f'{overall_score:.2f}%', f'{correlation_metrics["correlation_r2"]:.3f}', 
                 f'{correlation_metrics["correlation_mae"]:.3f}', 
                 f'{np.mean(mean_errors):.2f}%']
    }
    
    summary_df = pd.DataFrame(summary_data)
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=summary_df.values, colLabels=summary_df.columns, 
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Overall Performance Summary')
    
    plt.tight_layout()
    plt.savefig('./plots/error_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return overall_score

def print_detailed_report(distribution_metrics, correlation_metrics, physics_metrics, overall_score):
    """Print detailed error and accuracy report"""
    
    print("=" * 80)
    print("PHYSICS-INFORMED FLUXGAN ERROR & ACCURACY ANALYSIS")
    print("=" * 80)
    
    print(f"\n🎯 OVERALL PERFORMANCE SCORE: {overall_score:.2f}%")
    
    print("\n" + "=" * 60)
    print("📊 DISTRIBUTION ACCURACY BY FEATURE")
    print("=" * 60)
    
    for feature, metrics in distribution_metrics.items():
        print(f"\n{feature}:")
        print(f"  Mean Error: {metrics['mean_error_%']:.2f}%")
        print(f"  Std Error: {metrics['std_error_%']:.2f}%")
        print(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
        print(f"  KS p-value: {metrics['ks_pvalue']:.4f}")
        print(f"  Wasserstein Distance: {metrics['wasserstein_distance']:.4f}")
    
    print("\n" + "=" * 60)
    print("🔗 CORRELATION MATRIX ACCURACY")
    print("=" * 60)
    
    print(f"Correlation MAE: {correlation_metrics['correlation_mae']:.4f}")
    print(f"Correlation MSE: {correlation_metrics['correlation_mse']:.4f}")
    print(f"Correlation R²: {correlation_metrics['correlation_r2']:.4f}")
    print(f"Correlation Pearson: {correlation_metrics['correlation_pearson']:.4f}")
    print(f"Correlation p-value: {correlation_metrics['correlation_pearson_p']:.4f}")
    
    print("\n" + "=" * 60)
    print("🔬 PHYSICS CONSTRAINT ACCURACY")
    print("=" * 60)
    
    for constraint, accuracy in physics_metrics.items():
        print(f"{constraint}: {accuracy:.2f}%")
    
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE ASSESSMENT")
    print("=" * 60)
    
    # Grade the performance
    if overall_score >= 90:
        grade = "A+ (Excellent)"
    elif overall_score >= 80:
        grade = "A (Very Good)"
    elif overall_score >= 70:
        grade = "B+ (Good)"
    elif overall_score >= 60:
        grade = "B (Satisfactory)"
    elif overall_score >= 50:
        grade = "C (Needs Improvement)"
    else:
        grade = "D (Poor)"
    
    print(f"Overall Grade: {grade}")
    
    # Identify strengths and weaknesses
    print("\n🏆 STRENGTHS:")
    strengths = []
    for constraint, accuracy in physics_metrics.items():
        if accuracy >= 95:
            strengths.append(f"{constraint}: {accuracy:.1f}%")
    
    if strengths:
        for strength in strengths:
            print(f"  ✅ {strength}")
    else:
        print("  No constraints with >95% accuracy")
    
    print("\n⚠️ AREAS FOR IMPROVEMENT:")
    improvements = []
    for constraint, accuracy in physics_metrics.items():
        if accuracy < 80:
            improvements.append(f"{constraint}: {accuracy:.1f}%")
    
    if improvements:
        for improvement in improvements:
            print(f"  🔧 {improvement}")
    else:
        print("  All constraints performing well (>80% accuracy)")

def main():
    print("🚀 Loading trained model and generating samples for error analysis...")
    
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
    print("🎲 Generating 10,000 samples for analysis...")
    generated_samples, enrichment_conditions = generate_samples(generator, num_samples=10000)
    
    # Denormalize samples
    data_min = checkpoint['data_min']
    data_max = checkpoint['data_max']
    generated_data = denormalize_samples(generated_samples, data_min, data_max)
    
    # Create DataFrames
    gen_df = pd.DataFrame(generated_data, columns=feature_cols)
    real_df = pd.DataFrame(real_data, columns=feature_cols)
    
    print("📊 Calculating error and accuracy metrics...")
    
    # Calculate all metrics
    distribution_metrics = calculate_distribution_metrics(real_data, generated_data, feature_cols)
    correlation_metrics = calculate_correlation_metrics(real_df, gen_df)
    physics_metrics = calculate_physics_accuracy(gen_df)
    
    # Create visualizations
    print("📈 Creating error and accuracy plots...")
    overall_score = create_error_plots(distribution_metrics, correlation_metrics, physics_metrics)
    
    # Print detailed report
    print_detailed_report(distribution_metrics, correlation_metrics, physics_metrics, overall_score)
    
    # Save detailed metrics to CSV
    print("\n💾 Saving detailed metrics...")
    
    # Distribution metrics
    dist_df = pd.DataFrame(distribution_metrics).T
    dist_df.to_csv('./plots/distribution_accuracy_metrics.csv')
    
    # Physics metrics
    physics_df = pd.DataFrame(list(physics_metrics.items()), columns=['Constraint', 'Accuracy_%'])
    physics_df.to_csv('./plots/physics_accuracy_metrics.csv', index=False)
    
    # Overall summary
    summary_data = {
        'Metric': ['Overall_Physics_Accuracy_%', 'Correlation_R2', 'Correlation_MAE', 'Mean_Distribution_Error_%'],
        'Value': [overall_score, correlation_metrics['correlation_r2'], 
                 correlation_metrics['correlation_mae'], 
                 np.mean([distribution_metrics[f]['mean_error_%'] for f in feature_cols])]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('./plots/overall_performance_summary.csv', index=False)
    
    print(f"\n✅ Error and accuracy analysis complete!")
    print(f"📁 Results saved in ./plots/ directory")
    print(f"🎯 Overall Performance Score: {overall_score:.2f}%")

if __name__ == "__main__":
    main() 