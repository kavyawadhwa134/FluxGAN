import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def load_real_data():
    """Load and process real neutron trajectory data."""
    real_df = pd.read_csv('data/Sheet.csv')
    real_coords = real_df[['x', 'y', 'z']].values
    norm_params = np.load('data/neutron_train_norm_params.npy', allow_pickle=True).item()
    return real_coords, norm_params

def load_improved_synthetic_data(csv_path):
    """Load improved synthetic trajectory data."""
    synthetic_df = pd.read_csv(csv_path)
    
    trajectories = []
    for traj_id in synthetic_df['trajectory_id'].unique():
        traj_data = synthetic_df[synthetic_df['trajectory_id'] == traj_id]
        coords = traj_data[['x', 'y', 'z']].values
        trajectories.append(coords)
    
    return trajectories, synthetic_df

def calculate_improved_accuracy_metrics(real_coords, synthetic_trajectories):
    """Calculate comprehensive accuracy metrics for improved model."""
    
    # Flatten synthetic trajectories
    synthetic_coords = np.vstack(synthetic_trajectories)
    
    print("="*80)
    print("IMPROVED NEUTRON TRAJECTORY GAN - ACCURACY EVALUATION")
    print("="*80)
    
    print(f"\nData Summary:")
    print(f"  Real data points: {len(real_coords)}")
    print(f"  Synthetic trajectories: {len(synthetic_trajectories)}")
    print(f"  Synthetic data points: {len(synthetic_coords)}")
    
    # 1. BASIC STATISTICAL COMPARISON
    print(f"\n{'='*20} STATISTICAL ACCURACY METRICS {'='*20}")
    
    accuracy_scores = {}
    
    for i, coord in enumerate(['X', 'Y', 'Z']):
        real_mean = np.mean(real_coords[:, i])
        real_std = np.std(real_coords[:, i])
        real_min = np.min(real_coords[:, i])
        real_max = np.max(real_coords[:, i])
        
        synth_mean = np.mean(synthetic_coords[:, i])
        synth_std = np.std(synthetic_coords[:, i])
        synth_min = np.min(synthetic_coords[:, i])
        synth_max = np.max(synthetic_coords[:, i])
        
        print(f"\n{coord}-Coordinate Analysis:")
        print(f"  Real    - Mean: {real_mean:8.4f}, Std: {real_std:8.4f}, Range: [{real_min:7.3f}, {real_max:7.3f}]")
        print(f"  Synthetic - Mean: {synth_mean:8.4f}, Std: {synth_std:8.4f}, Range: [{synth_min:7.3f}, {synth_max:7.3f}]")
        
        # Calculate accuracy scores
        mean_accuracy = max(0, 1 - abs(real_mean - synth_mean) / (abs(real_mean) + 1e-8))
        std_accuracy = max(0, 1 - abs(real_std - synth_std) / (real_std + 1e-8))
        range_real = real_max - real_min
        range_synth = synth_max - synth_min
        range_accuracy = max(0, 1 - abs(range_real - range_synth) / (range_real + 1e-8))
        
        coord_accuracy = (mean_accuracy * 0.4 + std_accuracy * 0.4 + range_accuracy * 0.2) * 100
        accuracy_scores[f'{coord}_accuracy'] = coord_accuracy
        
        print(f"  Mean Accuracy:  {mean_accuracy*100:6.1f}%")
        print(f"  Std Accuracy:   {std_accuracy*100:6.1f}%") 
        print(f"  Range Accuracy: {range_accuracy*100:6.1f}%")
        print(f"  Overall {coord} Accuracy: {coord_accuracy:6.1f}%")
    
    # 2. STATISTICAL TESTS
    print(f"\n{'='*20} STATISTICAL SIGNIFICANCE TESTS {'='*20}")
    
    test_scores = {}
    for i, coord in enumerate(['X', 'Y', 'Z']):
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(real_coords[:, i], synthetic_coords[:, i])
        
        # Mann-Whitney U test (non-parametric)
        try:
            mw_stat, mw_pvalue = stats.mannwhitneyu(real_coords[:, i], synthetic_coords[:, i], alternative='two-sided')
        except:
            mw_pvalue = 0
        
        print(f"\n{coord}-Coordinate Statistical Tests:")
        print(f"  Kolmogorov-Smirnov: statistic={ks_stat:.4f}, p-value={ks_pvalue:.6f}")
        print(f"  Mann-Whitney U:     p-value={mw_pvalue:.6f}")
        
        # Higher p-value means more similar distributions
        ks_score = min(ks_pvalue / 0.05, 1.0) * 100  # Normalize to 100%
        mw_score = min(mw_pvalue / 0.05, 1.0) * 100
        
        test_scores[f'{coord}_ks_score'] = ks_score
        test_scores[f'{coord}_mw_score'] = mw_score
        
        print(f"  KS Similarity Score: {ks_score:6.1f}%")
        print(f"  MW Similarity Score: {mw_score:6.1f}%")
    
    # 3. PHYSICAL REALISM METRICS
    print(f"\n{'='*20} PHYSICAL REALISM METRICS {'='*20}")
    
    # Calculate step sizes for real data
    real_steps = []
    for i in range(len(real_coords) - 1):
        step = np.linalg.norm(real_coords[i+1] - real_coords[i])
        real_steps.append(step)
    
    # Calculate step sizes for synthetic data
    synth_steps = []
    for traj in synthetic_trajectories:
        for i in range(len(traj) - 1):
            step = np.linalg.norm(traj[i+1] - traj[i])
            synth_steps.append(step)
    
    real_step_mean = np.mean(real_steps)
    real_step_std = np.std(real_steps)
    synth_step_mean = np.mean(synth_steps)
    synth_step_std = np.std(synth_steps)
    
    print(f"\nStep Size Analysis:")
    print(f"  Real Steps      - Mean: {real_step_mean:.4f}, Std: {real_step_std:.4f}")
    print(f"  Synthetic Steps - Mean: {synth_step_mean:.4f}, Std: {synth_step_std:.4f}")
    
    step_mean_accuracy = max(0, 1 - abs(real_step_mean - synth_step_mean) / real_step_mean) * 100
    step_std_accuracy = max(0, 1 - abs(real_step_std - synth_step_std) / real_step_std) * 100
    
    print(f"  Step Mean Accuracy: {step_mean_accuracy:6.1f}%")
    print(f"  Step Std Accuracy:  {step_std_accuracy:6.1f}%")
    
    # 4. OVERALL ACCURACY CALCULATION
    print(f"\n{'='*20} OVERALL ACCURACY ASSESSMENT {'='*20}")
    
    # Coordinate accuracy (40% weight)
    coord_score = (accuracy_scores['X_accuracy'] + accuracy_scores['Y_accuracy'] + accuracy_scores['Z_accuracy']) / 3
    
    # Statistical test score (30% weight)
    stat_score = (np.mean([test_scores[f'{coord}_ks_score'] for coord in ['X', 'Y', 'Z']]) + 
                  np.mean([test_scores[f'{coord}_mw_score'] for coord in ['X', 'Y', 'Z']])) / 2
    
    # Physical realism score (30% weight)
    physics_score = (step_mean_accuracy + step_std_accuracy) / 2
    
    # Overall weighted score
    overall_accuracy = (coord_score * 0.4 + stat_score * 0.3 + physics_score * 0.3)
    
    print(f"\nComponent Scores:")
    print(f"  Coordinate Statistics Score: {coord_score:6.1f}% (weight: 40%)")
    print(f"  Statistical Tests Score:     {stat_score:6.1f}% (weight: 30%)")
    print(f"  Physical Realism Score:      {physics_score:6.1f}% (weight: 30%)")
    print(f"  " + "="*50)
    print(f"  OVERALL ACCURACY SCORE:      {overall_accuracy:6.1f}%")
    
    # 5. INTERPRETATION
    print(f"\n{'='*20} PERFORMANCE INTERPRETATION {'='*20}")
    
    if overall_accuracy >= 80:
        performance = "EXCELLENT"
        interpretation = "Synthetic trajectories closely match real neutron data"
    elif overall_accuracy >= 65:
        performance = "VERY GOOD"
        interpretation = "Synthetic trajectories show strong similarity to real data"
    elif overall_accuracy >= 50:
        performance = "GOOD"
        interpretation = "Synthetic trajectories capture main patterns well"
    elif overall_accuracy >= 35:
        performance = "FAIR"
        interpretation = "Synthetic trajectories show reasonable similarity"
    else:
        performance = "NEEDS IMPROVEMENT"
        interpretation = "Synthetic trajectories need further refinement"
    
    print(f"\nPerformance Level: {performance}")
    print(f"Interpretation: {interpretation}")
    
    # 6. DETAILED BREAKDOWN BY COORDINATE
    print(f"\n{'='*20} COORDINATE-WISE BREAKDOWN {'='*20}")
    
    best_coord = max(['X', 'Y', 'Z'], key=lambda c: accuracy_scores[f'{c}_accuracy'])
    worst_coord = min(['X', 'Y', 'Z'], key=lambda c: accuracy_scores[f'{c}_accuracy'])
    
    print(f"Best Performance:  {best_coord}-coordinate ({accuracy_scores[f'{best_coord}_accuracy']:.1f}%)")
    print(f"Worst Performance: {worst_coord}-coordinate ({accuracy_scores[f'{worst_coord}_accuracy']:.1f}%)")
    
    if accuracy_scores['Z_accuracy'] > 80:
        print("✅ Z-coordinate (depth) shows excellent accuracy - critical for reactor physics!")
    
    print("="*80)
    
    return overall_accuracy

def create_comparison_plots(real_coords, synthetic_trajectories):
    """Create comprehensive comparison plots."""
    
    synthetic_coords = np.vstack(synthetic_trajectories)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Distribution plots
    for i, (coord, color) in enumerate(zip(['X', 'Y', 'Z'], ['red', 'green', 'blue'])):
        ax = axes[0, i]
        
        ax.hist(real_coords[:, i], bins=30, alpha=0.7, label='Real', density=True, color='lightblue', edgecolor='blue')
        ax.hist(synthetic_coords[:, i], bins=30, alpha=0.7, label='Synthetic', density=True, color='lightcoral', edgecolor='red')
        ax.set_xlabel(f'{coord} Coordinate')
        ax.set_ylabel('Density')
        ax.set_title(f'{coord} Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Box plots
    for i, coord in enumerate(['X', 'Y', 'Z']):
        ax = axes[1, i]
        
        data_to_plot = [real_coords[:, i], synthetic_coords[:, i]]
        box_plot = ax.boxplot(data_to_plot, labels=['Real', 'Synthetic'], patch_artist=True)
        
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_ylabel(f'{coord} Coordinate')
        ax.set_title(f'{coord} Box Plot Comparison')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neutron_results_v2/improved_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run comprehensive accuracy evaluation for improved model."""
    
    print("Loading data for improved model evaluation...")
    
    # Load real data
    real_coords, norm_params = load_real_data()
    
    # Load improved synthetic data
    synthetic_trajectories, synthetic_df = load_improved_synthetic_data(
        'neutron_results_v2/improved_synthetic_trajectories_epoch_450.csv'
    )
    
    # Calculate accuracy
    overall_accuracy = calculate_improved_accuracy_metrics(real_coords, synthetic_trajectories)
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(real_coords, synthetic_trajectories)
    
    print(f"\nComparison plots saved to: neutron_results_v2/improved_accuracy_comparison.png")
    print(f"\nFINAL ACCURACY SCORE: {overall_accuracy:.1f}%")
    
    return overall_accuracy

if __name__ == "__main__":
    accuracy_score = main()