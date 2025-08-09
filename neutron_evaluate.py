import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

def load_real_data():
    """Load and process real neutron trajectory data."""
    # Load original CSV
    real_df = pd.read_csv('data/Sheet.csv')
    real_coords = real_df[['x', 'y', 'z']].values
    
    # Load normalization parameters
    norm_params = np.load('data/neutron_train_norm_params.npy', allow_pickle=True).item()
    
    return real_coords, norm_params

def load_synthetic_data(csv_path):
    """Load synthetic trajectory data."""
    synthetic_df = pd.read_csv(csv_path)
    
    # Group by trajectory_id to get individual trajectories
    trajectories = []
    for traj_id in synthetic_df['trajectory_id'].unique():
        traj_data = synthetic_df[synthetic_df['trajectory_id'] == traj_id]
        coords = traj_data[['x', 'y', 'z']].values
        trajectories.append(coords)
    
    return trajectories, synthetic_df

def calculate_trajectory_statistics(trajectories):
    """Calculate statistical properties of trajectories."""
    stats_dict = {}
    
    all_coords = np.vstack(trajectories)
    
    # Basic statistics
    stats_dict['mean_x'] = np.mean(all_coords[:, 0])
    stats_dict['mean_y'] = np.mean(all_coords[:, 1])
    stats_dict['mean_z'] = np.mean(all_coords[:, 2])
    
    stats_dict['std_x'] = np.std(all_coords[:, 0])
    stats_dict['std_y'] = np.std(all_coords[:, 1])
    stats_dict['std_z'] = np.std(all_coords[:, 2])
    
    # Range statistics
    stats_dict['range_x'] = np.max(all_coords[:, 0]) - np.min(all_coords[:, 0])
    stats_dict['range_y'] = np.max(all_coords[:, 1]) - np.min(all_coords[:, 1])
    stats_dict['range_z'] = np.max(all_coords[:, 2]) - np.min(all_coords[:, 2])
    
    # Trajectory-specific statistics
    trajectory_lengths = [len(traj) for traj in trajectories]
    stats_dict['avg_trajectory_length'] = np.mean(trajectory_lengths)
    stats_dict['std_trajectory_length'] = np.std(trajectory_lengths)
    
    # Step sizes (distances between consecutive points)
    step_sizes = []
    for traj in trajectories:
        if len(traj) > 1:
            steps = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
            step_sizes.extend(steps)
    
    if step_sizes:
        stats_dict['avg_step_size'] = np.mean(step_sizes)
        stats_dict['std_step_size'] = np.std(step_sizes)
    else:
        stats_dict['avg_step_size'] = 0
        stats_dict['std_step_size'] = 0
    
    return stats_dict

def calculate_physical_metrics(trajectories):
    """Calculate physics-based metrics for neutron trajectories."""
    metrics = {}
    
    all_step_sizes = []
    all_direction_changes = []
    all_curvatures = []
    
    for traj in trajectories:
        if len(traj) < 3:
            continue
            
        # Step sizes
        steps = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
        all_step_sizes.extend(steps)
        
        # Direction changes (angle between consecutive segments)
        for i in range(len(traj) - 2):
            v1 = traj[i+1] - traj[i]
            v2 = traj[i+2] - traj[i+1]
            
            # Avoid division by zero
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
                angle = np.arccos(cos_angle)
                all_direction_changes.append(angle)
        
        # Curvature (simplified metric)
        if len(traj) >= 3:
            for i in range(1, len(traj) - 1):
                p1, p2, p3 = traj[i-1], traj[i], traj[i+1]
                
                # Calculate curvature using the formula for discrete points
                v1 = p2 - p1
                v2 = p3 - p2
                
                cross_product = np.cross(v1, v2)
                if isinstance(cross_product, np.ndarray):
                    curvature = np.linalg.norm(cross_product)
                else:
                    curvature = abs(cross_product)
                
                v1_norm = np.linalg.norm(v1)
                if v1_norm > 1e-8:
                    curvature /= v1_norm**3
                    all_curvatures.append(curvature)
    
    metrics['step_size_mean'] = np.mean(all_step_sizes) if all_step_sizes else 0
    metrics['step_size_std'] = np.std(all_step_sizes) if all_step_sizes else 0
    metrics['direction_change_mean'] = np.mean(all_direction_changes) if all_direction_changes else 0
    metrics['direction_change_std'] = np.std(all_direction_changes) if all_direction_changes else 0
    metrics['curvature_mean'] = np.mean(all_curvatures) if all_curvatures else 0
    metrics['curvature_std'] = np.std(all_curvatures) if all_curvatures else 0
    
    return metrics

def statistical_tests(real_coords, synthetic_coords):
    """Perform statistical tests to compare real and synthetic data."""
    tests = {}
    
    # Kolmogorov-Smirnov test for each dimension
    for i, dim in enumerate(['x', 'y', 'z']):
        ks_stat, ks_pvalue = stats.ks_2samp(real_coords[:, i], synthetic_coords[:, i])
        tests[f'ks_{dim}_statistic'] = ks_stat
        tests[f'ks_{dim}_pvalue'] = ks_pvalue
        
        # Anderson-Darling test
        try:
            combined = np.concatenate([real_coords[:, i], synthetic_coords[:, i]])
            labels = np.concatenate([np.zeros(len(real_coords)), np.ones(len(synthetic_coords))])
            ad_stat, ad_critical, ad_pvalue = stats.anderson_ksamp([real_coords[:, i], synthetic_coords[:, i]])
            tests[f'ad_{dim}_statistic'] = ad_stat
            tests[f'ad_{dim}_pvalue'] = ad_pvalue if ad_pvalue is not None else 0
        except:
            tests[f'ad_{dim}_statistic'] = np.nan
            tests[f'ad_{dim}_pvalue'] = np.nan
    
    return tests

def create_evaluation_plots(real_coords, synthetic_trajectories, save_path="neutron_results"):
    """Create comprehensive evaluation plots."""
    
    # Flatten synthetic trajectories
    synthetic_coords = np.vstack(synthetic_trajectories)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Distribution comparisons
    for i, (dim, label) in enumerate(zip([0, 1, 2], ['X', 'Y', 'Z'])):
        ax = plt.subplot(3, 4, i+1)
        
        plt.hist(real_coords[:, i], bins=30, alpha=0.7, label='Real', density=True, color='blue')
        plt.hist(synthetic_coords[:, i], bins=30, alpha=0.7, label='Synthetic', density=True, color='red')
        plt.xlabel(f'{label} Coordinate')
        plt.ylabel('Density')
        plt.title(f'{label} Coordinate Distribution')
        plt.legend()
    
    # 2. Q-Q plots
    for i, (dim, label) in enumerate(zip([0, 1, 2], ['X', 'Y', 'Z'])):
        ax = plt.subplot(3, 4, i+5)
        
        stats.probplot(real_coords[:, i], plot=plt, dist="norm")
        plt.title(f'Q-Q Plot: Real {label}')
        
        ax = plt.subplot(3, 4, i+6)
        stats.probplot(synthetic_coords[:, i], plot=plt, dist="norm")
        plt.title(f'Q-Q Plot: Synthetic {label}')
    
    # 3. Scatter plots
    ax = plt.subplot(3, 4, 9)
    plt.scatter(real_coords[:, 0], real_coords[:, 1], alpha=0.6, label='Real', s=20)
    plt.scatter(synthetic_coords[:, 0], synthetic_coords[:, 1], alpha=0.6, label='Synthetic', s=20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('X vs Y Scatter Plot')
    plt.legend()
    
    ax = plt.subplot(3, 4, 10)
    plt.scatter(real_coords[:, 0], real_coords[:, 2], alpha=0.6, label='Real', s=20)
    plt.scatter(synthetic_coords[:, 0], synthetic_coords[:, 2], alpha=0.6, label='Synthetic', s=20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.title('X vs Z Scatter Plot')
    plt.legend()
    
    ax = plt.subplot(3, 4, 11)
    plt.scatter(real_coords[:, 1], real_coords[:, 2], alpha=0.6, label='Real', s=20)
    plt.scatter(synthetic_coords[:, 1], synthetic_coords[:, 2], alpha=0.6, label='Synthetic', s=20)
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.title('Y vs Z Scatter Plot')
    plt.legend()
    
    # 4. 3D trajectory comparison
    ax = plt.subplot(3, 4, 12, projection='3d')
    
    # Plot a few real trajectory segments
    if len(real_coords) > 50:
        step = len(real_coords) // 5  # Show 5 segments
        for i in range(0, len(real_coords)-step, step):
            segment = real_coords[i:i+step]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], 'b-', alpha=0.7, linewidth=1)
    
    # Plot synthetic trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(synthetic_trajectories), 10)))
    for i, traj in enumerate(synthetic_trajectories[:10]):  # Show first 10
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[i], alpha=0.7, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Comparison')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/accuracy_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_accuracy_report(real_coords, synthetic_trajectories, csv_path):
    """Generate a comprehensive accuracy report."""
    
    print("="*80)
    print("NEUTRON TRAJECTORY GAN - ACCURACY EVALUATION REPORT")
    print("="*80)
    
    # Convert single trajectory to list format for consistency
    real_trajectories = [real_coords]
    synthetic_coords = np.vstack(synthetic_trajectories)
    
    # Calculate statistics
    real_stats = calculate_trajectory_statistics(real_trajectories)
    synthetic_stats = calculate_trajectory_statistics(synthetic_trajectories)
    
    print("\n1. BASIC STATISTICS COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<25} {'Real Data':<15} {'Synthetic':<15} {'Difference':<15}")
    print("-" * 70)
    
    for key in ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']:
        real_val = real_stats[key]
        synth_val = synthetic_stats[key]
        diff = abs(real_val - synth_val)
        print(f"{key:<25} {real_val:<15.4f} {synth_val:<15.4f} {diff:<15.4f}")
    
    # Physical metrics
    print("\n2. PHYSICAL METRICS COMPARISON")
    print("-" * 50)
    real_physics = calculate_physical_metrics(real_trajectories)
    synthetic_physics = calculate_physical_metrics(synthetic_trajectories)
    
    print(f"{'Metric':<25} {'Real Data':<15} {'Synthetic':<15} {'Difference':<15}")
    print("-" * 70)
    
    for key in real_physics:
        real_val = real_physics[key]
        synth_val = synthetic_physics[key]
        diff = abs(real_val - synth_val)
        print(f"{key:<25} {real_val:<15.4f} {synth_val:<15.4f} {diff:<15.4f}")
    
    # Statistical tests
    print("\n3. STATISTICAL TESTS")
    print("-" * 50)
    tests = statistical_tests(real_coords, synthetic_coords)
    
    print(f"{'Test':<30} {'Statistic':<15} {'P-value':<15} {'Significant?':<15}")
    print("-" * 75)
    
    for dim in ['x', 'y', 'z']:
        ks_stat = tests[f'ks_{dim}_statistic']
        ks_pval = tests[f'ks_{dim}_pvalue']
        significant = "Yes" if ks_pval < 0.05 else "No"
        print(f"KS Test ({dim}):{'':<18} {ks_stat:<15.4f} {ks_pval:<15.4f} {significant:<15}")
    
    # Overall accuracy score
    print("\n4. OVERALL ACCURACY ASSESSMENT")
    print("-" * 50)
    
    # Calculate accuracy score based on multiple factors
    stat_score = 0
    coord_weights = {'mean': 0.3, 'std': 0.3, 'range': 0.2}
    
    for coord in ['x', 'y', 'z']:
        for stat_type in ['mean', 'std']:
            key = f"{stat_type}_{coord}"
            real_val = real_stats[key]
            synth_val = synthetic_stats[key]
            
            if real_val != 0:
                relative_error = abs(real_val - synth_val) / abs(real_val)
                accuracy = max(0, 1 - relative_error)
                stat_score += accuracy * coord_weights[stat_type] / 3  # Divide by 3 for x,y,z
    
    # Physics score
    physics_score = 0
    physics_weights = {'step_size_mean': 0.4, 'direction_change_mean': 0.3, 'curvature_mean': 0.3}
    
    for metric in ['step_size_mean', 'direction_change_mean', 'curvature_mean']:
        real_val = real_physics[metric]
        synth_val = synthetic_physics[metric]
        
        if real_val != 0:
            relative_error = abs(real_val - synth_val) / abs(real_val)
            accuracy = max(0, 1 - relative_error)
            physics_score += accuracy * physics_weights[metric]
    
    # Statistical test score (based on p-values)
    test_score = 0
    for dim in ['x', 'y', 'z']:
        p_val = tests[f'ks_{dim}_pvalue']
        # Higher p-value means distributions are more similar
        test_score += min(p_val, 0.05) / 0.05 / 3  # Normalize and average
    
    # Overall score
    overall_score = (stat_score * 0.4 + physics_score * 0.4 + test_score * 0.2) * 100
    
    print(f"Statistical Similarity Score: {stat_score * 100:.1f}%")
    print(f"Physical Realism Score:      {physics_score * 100:.1f}%")
    print(f"Distribution Match Score:    {test_score * 100:.1f}%")
    print("-" * 50)
    print(f"OVERALL ACCURACY SCORE:      {overall_score:.1f}%")
    
    # Interpretation
    print(f"\n5. INTERPRETATION")
    print("-" * 50)
    if overall_score >= 80:
        print("EXCELLENT: Synthetic trajectories closely match real data")
    elif overall_score >= 70:
        print("GOOD: Synthetic trajectories show strong similarity to real data")
    elif overall_score >= 60:
        print("FAIR: Synthetic trajectories capture main patterns but need improvement")
    else:
        print("POOR: Synthetic trajectories need significant improvement")
    
    print(f"\nDetailed evaluation plots saved to: neutron_results/accuracy_evaluation.png")
    print("="*80)
    
    return overall_score

def main():
    """Main evaluation function."""
    
    # Load data
    print("Loading real neutron trajectory data...")
    real_coords, norm_params = load_real_data()
    
    print("Loading synthetic trajectory data...")
    synthetic_trajectories, synthetic_df = load_synthetic_data('neutron_results/synthetic_trajectories_epoch_450.csv')
    
    print(f"Real data: {len(real_coords)} points")
    print(f"Synthetic data: {len(synthetic_trajectories)} trajectories, {sum(len(t) for t in synthetic_trajectories)} total points")
    
    # Generate evaluation plots
    print("\nGenerating evaluation plots...")
    create_evaluation_plots(real_coords, synthetic_trajectories)
    
    # Generate comprehensive report
    print("\nGenerating accuracy report...")
    accuracy_score = generate_accuracy_report(real_coords, synthetic_trajectories, 
                                            'neutron_results/synthetic_trajectories_epoch_450.csv')
    
    return accuracy_score

if __name__ == "__main__":
    accuracy_score = main()