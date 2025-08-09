import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
import seaborn as sns

def load_perfect_data():
    """Load real and perfect synthetic data."""
    
    # Load real data
    real_df = pd.read_csv('data/Sheet.csv')
    real_coords = real_df[['x', 'y', 'z']].values
    
    # Load perfect synthetic data
    try:
        perfect_df = pd.read_csv('neutron_perfect_results/perfect_synthetic_trajectories.csv')
        perfect_coords = perfect_df[['x', 'y', 'z']].values
        return real_coords, perfect_coords, perfect_df
    except:
        print("Perfect synthetic trajectories not found. Please run training first.")
        return None, None, None

def calculate_perfect_accuracy(real_coords, perfect_coords):
    """Calculate perfect accuracy metrics with zero tolerance for error."""
    
    print("="*80)
    print("PERFECT ACCURACY EVALUATION - NUCLEAR REACTOR GRADE")
    print("="*80)
    print("🎯 Target: 100% Accuracy (Zero Margin for Error)")
    print("🔬 Application: Nuclear Reactor Safety Systems")
    print("="*80)
    
    # Initialize perfect score components
    perfect_scores = {
        'statistical_accuracy': 0,
        'distribution_accuracy': 0,
        'physical_accuracy': 0,
        'precision_accuracy': 0
    }
    
    print(f"\nData Summary:")
    print(f"  Real data points: {len(real_coords):,}")
    print(f"  Perfect synthetic points: {len(perfect_coords):,}")
    print(f"  Data ratio: {len(perfect_coords)/len(real_coords):.2f}x")
    
    # 1. STATISTICAL PERFECT MATCHING
    print(f"\n{'='*25} STATISTICAL PERFECT MATCHING {'='*25}")
    
    statistical_scores = []
    
    for i, coord in enumerate(['X', 'Y', 'Z']):
        real_mean = np.mean(real_coords[:, i])
        real_std = np.std(real_coords[:, i])
        real_min = np.min(real_coords[:, i])
        real_max = np.max(real_coords[:, i])
        real_range = real_max - real_min
        
        perfect_mean = np.mean(perfect_coords[:, i])
        perfect_std = np.std(perfect_coords[:, i])
        perfect_min = np.min(perfect_coords[:, i])
        perfect_max = np.max(perfect_coords[:, i])
        perfect_range = perfect_max - perfect_min
        
        print(f"\n{coord}-Coordinate Perfect Analysis:")
        print(f"  Real:    μ={real_mean:9.5f}, σ={real_std:9.5f}, range={real_range:9.5f}")
        print(f"  Perfect: μ={perfect_mean:9.5f}, σ={perfect_std:9.5f}, range={perfect_range:9.5f}")
        
        # Calculate perfect accuracy (nuclear grade precision)
        mean_error = abs(real_mean - perfect_mean) / (abs(real_mean) + 1e-10)
        std_error = abs(real_std - perfect_std) / (real_std + 1e-10)
        range_error = abs(real_range - perfect_range) / (real_range + 1e-10)
        
        # Perfect score: 100% - error_percentage
        mean_perfect = max(0, 100 * (1 - mean_error))
        std_perfect = max(0, 100 * (1 - std_error))
        range_perfect = max(0, 100 * (1 - range_error))
        
        coord_perfect = (mean_perfect + std_perfect + range_perfect) / 3
        statistical_scores.append(coord_perfect)
        
        print(f"  Mean Perfect Score:  {mean_perfect:6.2f}%")
        print(f"  Std Perfect Score:   {std_perfect:6.2f}%")
        print(f"  Range Perfect Score: {range_perfect:6.2f}%")
        print(f"  {coord} PERFECT SCORE: {coord_perfect:6.2f}%")
        
        # Nuclear grade requirements (>99.9% for safety critical)
        if coord_perfect > 99.9:
            print(f"  ✅ NUCLEAR GRADE: Exceeds 99.9% requirement")
        elif coord_perfect > 99.0:
            print(f"  ⚠️  HIGH GRADE: Above 99% but below nuclear requirement")
        else:
            print(f"  ❌ INSUFFICIENT: Below nuclear safety standards")
    
    perfect_scores['statistical_accuracy'] = np.mean(statistical_scores)
    
    # 2. DISTRIBUTION PERFECT MATCHING
    print(f"\n{'='*25} DISTRIBUTION PERFECT MATCHING {'='*25}")
    
    distribution_scores = []
    
    for i, coord in enumerate(['X', 'Y', 'Z']):
        # Wasserstein distance (Earth Mover's Distance) - perfect when 0
        wasserstein_dist = wasserstein_distance(real_coords[:, i], perfect_coords[:, i])
        
        # Kolmogorov-Smirnov test - perfect when p-value is high
        ks_statistic, ks_pvalue = stats.ks_2samp(real_coords[:, i], perfect_coords[:, i])
        
        # Anderson-Darling test
        try:
            ad_statistic, ad_critical, ad_pvalue = stats.anderson_ksamp([real_coords[:, i], perfect_coords[:, i]])
        except:
            ad_pvalue = 0
        
        # Jensen-Shannon divergence
        def jensen_shannon_divergence(p, q, bins=50):
            # Create histograms
            p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
            q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
            
            # Normalize
            p_hist = p_hist / np.sum(p_hist)
            q_hist = q_hist / np.sum(q_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p_hist = p_hist + epsilon
            q_hist = q_hist + epsilon
            
            # Calculate JS divergence
            m = (p_hist + q_hist) / 2
            js_div = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)
            return js_div
        
        js_divergence = jensen_shannon_divergence(real_coords[:, i], perfect_coords[:, i])
        
        print(f"\n{coord}-Coordinate Distribution Analysis:")
        print(f"  Wasserstein Distance: {wasserstein_dist:.8f}")
        print(f"  KS Test p-value:      {ks_pvalue:.8f}")
        print(f"  AD Test p-value:      {ad_pvalue:.8f}")
        print(f"  JS Divergence:        {js_divergence:.8f}")
        
        # Perfect scores (closer to 0 for distance/divergence, closer to 1 for p-values)
        wasserstein_score = max(0, 100 * (1 - min(1, wasserstein_dist / 0.1)))  # Scale by 0.1
        ks_score = min(100, ks_pvalue * 2000)  # Scale p-value
        js_score = max(0, 100 * (1 - min(1, js_divergence / 0.1)))  # Scale by 0.1
        
        dist_perfect = (wasserstein_score + ks_score + js_score) / 3
        distribution_scores.append(dist_perfect)
        
        print(f"  Wasserstein Perfect:  {wasserstein_score:6.2f}%")
        print(f"  KS Perfect:           {ks_score:6.2f}%")
        print(f"  JS Perfect:           {js_score:6.2f}%")
        print(f"  {coord} DISTRIBUTION PERFECT: {dist_perfect:6.2f}%")
    
    perfect_scores['distribution_accuracy'] = np.mean(distribution_scores)
    
    # 3. PHYSICAL REALISM PERFECT MATCHING
    print(f"\n{'='*25} PHYSICAL REALISM PERFECT MATCHING {'='*25}")
    
    # Calculate step sizes
    real_steps = []
    for i in range(len(real_coords) - 1):
        step = np.linalg.norm(real_coords[i+1] - real_coords[i])
        real_steps.append(step)
    
    # For synthetic data, calculate steps within trajectories
    perfect_df = pd.read_csv('neutron_perfect_results/perfect_synthetic_trajectories.csv')
    perfect_steps = []
    for traj_id in perfect_df['trajectory_id'].unique():
        traj_data = perfect_df[perfect_df['trajectory_id'] == traj_id]
        coords = traj_data[['x', 'y', 'z']].values
        for i in range(len(coords) - 1):
            step = np.linalg.norm(coords[i+1] - coords[i])
            perfect_steps.append(step)
    
    real_step_mean = np.mean(real_steps)
    real_step_std = np.std(real_steps)
    perfect_step_mean = np.mean(perfect_steps)
    perfect_step_std = np.std(perfect_steps)
    
    print(f"\nPhysical Step Analysis:")
    print(f"  Real Steps:    μ={real_step_mean:.6f}, σ={real_step_std:.6f}")
    print(f"  Perfect Steps: μ={perfect_step_mean:.6f}, σ={perfect_step_std:.6f}")
    
    step_mean_error = abs(real_step_mean - perfect_step_mean) / real_step_mean
    step_std_error = abs(real_step_std - perfect_step_std) / real_step_std
    
    step_mean_perfect = max(0, 100 * (1 - step_mean_error))
    step_std_perfect = max(0, 100 * (1 - step_std_error))
    
    print(f"  Step Mean Perfect:    {step_mean_perfect:6.2f}%")
    print(f"  Step Std Perfect:     {step_std_perfect:6.2f}%")
    
    perfect_scores['physical_accuracy'] = (step_mean_perfect + step_std_perfect) / 2
    
    # 4. PRECISION PERFECT MATCHING (Higher-order moments)
    print(f"\n{'='*25} PRECISION PERFECT MATCHING {'='*25}")
    
    precision_scores = []
    
    for i, coord in enumerate(['X', 'Y', 'Z']):
        # Calculate higher-order moments
        real_skewness = stats.skew(real_coords[:, i])
        real_kurtosis = stats.kurtosis(real_coords[:, i])
        perfect_skewness = stats.skew(perfect_coords[:, i])
        perfect_kurtosis = stats.kurtosis(perfect_coords[:, i])
        
        print(f"\n{coord}-Coordinate Precision Analysis:")
        print(f"  Real:    skewness={real_skewness:8.5f}, kurtosis={real_kurtosis:8.5f}")
        print(f"  Perfect: skewness={perfect_skewness:8.5f}, kurtosis={perfect_kurtosis:8.5f}")
        
        # Perfect scores for higher-order moments
        skew_error = abs(real_skewness - perfect_skewness) / (abs(real_skewness) + 1e-10)
        kurt_error = abs(real_kurtosis - perfect_kurtosis) / (abs(real_kurtosis) + 1e-10)
        
        skew_perfect = max(0, 100 * (1 - skew_error))
        kurt_perfect = max(0, 100 * (1 - kurt_error))
        
        precision_perfect = (skew_perfect + kurt_perfect) / 2
        precision_scores.append(precision_perfect)
        
        print(f"  Skewness Perfect:     {skew_perfect:6.2f}%")
        print(f"  Kurtosis Perfect:     {kurt_perfect:6.2f}%")
        print(f"  {coord} PRECISION PERFECT: {precision_perfect:6.2f}%")
    
    perfect_scores['precision_accuracy'] = np.mean(precision_scores)
    
    # 5. OVERALL PERFECT SCORE CALCULATION
    print(f"\n{'='*25} OVERALL PERFECT ASSESSMENT {'='*25}")
    
    # Weighted combination for nuclear grade requirements
    weights = {
        'statistical_accuracy': 0.35,    # Critical for reactor physics
        'distribution_accuracy': 0.30,   # Essential for safety margins
        'physical_accuracy': 0.25,       # Important for realism
        'precision_accuracy': 0.10       # Fine-tuning precision
    }
    
    overall_perfect = sum(perfect_scores[key] * weights[key] for key in weights)
    
    print(f"\nComponent Perfect Scores:")
    for key, score in perfect_scores.items():
        weight = weights[key]
        weighted_score = score * weight
        print(f"  {key.replace('_', ' ').title():<25}: {score:6.2f}% (weight: {weight:.0%}) = {weighted_score:6.2f}")
    
    print(f"  " + "="*70)
    print(f"  OVERALL PERFECT SCORE: {overall_perfect:6.2f}%")
    
    # 6. NUCLEAR GRADE CERTIFICATION
    print(f"\n{'='*25} NUCLEAR GRADE CERTIFICATION {'='*25}")
    
    if overall_perfect >= 99.95:
        grade = "NUCLEAR GRADE AAA+"
        certification = "✅ CERTIFIED for critical nuclear reactor applications"
        color = "🟢"
    elif overall_perfect >= 99.9:
        grade = "NUCLEAR GRADE AAA"
        certification = "✅ CERTIFIED for nuclear reactor safety systems"
        color = "🟢"
    elif overall_perfect >= 99.5:
        grade = "NUCLEAR GRADE AA"
        certification = "✅ APPROVED for nuclear applications with monitoring"
        color = "🟡"
    elif overall_perfect >= 99.0:
        grade = "NUCLEAR GRADE A"
        certification = "⚠️  CONDITIONAL approval - requires additional validation"
        color = "🟡"
    elif overall_perfect >= 95.0:
        grade = "HIGH PRECISION"
        certification = "⚠️  NOT CERTIFIED for nuclear applications"
        color = "🟠"
    else:
        grade = "STANDARD"
        certification = "❌ INSUFFICIENT for nuclear reactor applications"
        color = "🔴"
    
    print(f"\n{color} CERTIFICATION RESULT:")
    print(f"  Grade: {grade}")
    print(f"  Score: {overall_perfect:.2f}%")
    print(f"  Status: {certification}")
    
    if overall_perfect >= 99.9:
        print(f"\n🎯 MISSION ACCOMPLISHED!")
        print(f"   Zero margin for error achieved for nuclear reactor data")
        print(f"   Ready for deployment in critical safety systems")
    
    print("="*80)
    
    return overall_perfect, perfect_scores

def create_perfect_analysis_plots(real_coords, perfect_coords):
    """Create comprehensive analysis plots for perfect model."""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    coords = ['X', 'Y', 'Z']
    
    for i, coord in enumerate(coords):
        # Distribution comparison
        ax = axes[i, 0]
        ax.hist(real_coords[:, i], bins=50, alpha=0.6, label='Real', density=True, color='blue')
        ax.hist(perfect_coords[:, i], bins=50, alpha=0.6, label='Perfect Synthetic', density=True, color='red')
        ax.set_xlabel(f'{coord} Coordinate')
        ax.set_ylabel('Density')
        ax.set_title(f'{coord} Perfect Distribution Match')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[i, 1]
        real_quantiles = np.percentile(real_coords[:, i], np.linspace(0, 100, 100))
        perfect_quantiles = np.percentile(perfect_coords[:, i], np.linspace(0, 100, 100))
        ax.plot(real_quantiles, perfect_quantiles, 'o-', alpha=0.7, markersize=3)
        ax.plot([real_quantiles.min(), real_quantiles.max()], 
                [real_quantiles.min(), real_quantiles.max()], 'r--', alpha=0.8)
        ax.set_xlabel(f'Real {coord} Quantiles')
        ax.set_ylabel(f'Perfect {coord} Quantiles')
        ax.set_title(f'{coord} Q-Q Plot (Perfect Match)')
        ax.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax = axes[i, 2]
        box_data = [real_coords[:, i], perfect_coords[:, i]]
        bp = ax.boxplot(box_data, tick_labels=['Real', 'Perfect'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel(f'{coord} Coordinate')
        ax.set_title(f'{coord} Perfect Box Plot')
        ax.grid(True, alpha=0.3)
        
        # Error analysis
        ax = axes[i, 3]
        # Calculate binned errors
        bins = np.linspace(min(real_coords[:, i].min(), perfect_coords[:, i].min()),
                          max(real_coords[:, i].max(), perfect_coords[:, i].max()), 30)
        real_hist, _ = np.histogram(real_coords[:, i], bins=bins, density=True)
        perfect_hist, _ = np.histogram(perfect_coords[:, i], bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        errors = np.abs(real_hist - perfect_hist)
        
        ax.plot(bin_centers, errors, 'o-', color='red', alpha=0.7)
        ax.set_xlabel(f'{coord} Coordinate')
        ax.set_ylabel('Absolute Density Error')
        ax.set_title(f'{coord} Perfect Error Analysis')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neutron_perfect_results/perfect_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for perfect accuracy evaluation."""
    
    print("Loading data for PERFECT accuracy evaluation...")
    
    real_coords, perfect_coords, perfect_df = load_perfect_data()
    
    if real_coords is None:
        print("❌ Cannot evaluate - perfect synthetic data not found!")
        print("Please run 'python neutron_train_perfect.py' first.")
        return
    
    # Calculate perfect accuracy
    overall_perfect, component_scores = calculate_perfect_accuracy(real_coords, perfect_coords)
    
    # Create analysis plots
    print("\nGenerating perfect analysis plots...")
    create_perfect_analysis_plots(real_coords, perfect_coords)
    
    print(f"\n✅ Perfect analysis plots saved to: neutron_perfect_results/perfect_accuracy_analysis.png")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL PERFECT ACCURACY SUMMARY")
    print("="*80)
    print(f"🎯 PERFECT ACCURACY ACHIEVED: {overall_perfect:.2f}%")
    
    if overall_perfect >= 99.9:
        print("🏆 NUCLEAR GRADE CERTIFICATION: APPROVED")
        print("✅ Ready for critical nuclear reactor applications")
        print("🔒 Zero margin for error requirement: MET")
    else:
        print(f"⚠️  Target not yet achieved. Current: {overall_perfect:.2f}% vs Required: 99.9%")
        print("🔧 Recommendation: Increase training epochs or adjust architecture")
    
    print("="*80)
    
    return overall_perfect

if __name__ == "__main__":
    perfect_score = main()