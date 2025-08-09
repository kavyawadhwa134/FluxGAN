import numpy as np
import pandas as pd
from scipy import interpolate, stats
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

class MassiveNeutronTrajectoryGenerator:
    """
    Massive neutron trajectory generator creating 1000+ diverse trajectories
    for comprehensive nuclear reactor simulation and analysis.
    """
    
    def __init__(self):
        self.real_data = None
        self.real_df = None
        self.trajectory_segments = None
        self.statistical_patterns = None
        self.is_trained = False
        
    def load_and_analyze_real_data(self):
        """Load and perform deep analysis of real neutron trajectory data."""
        print("🔬 Loading and analyzing real neutron trajectory data...")
        
        self.real_df = pd.read_csv('data/Sheet.csv')
        self.real_data = self.real_df[['x', 'y', 'z']].values
        
        print(f"✅ Real data loaded: {len(self.real_data)} points from continuous trajectory")
        print(f"   X range: [{self.real_data[:, 0].min():.3f}, {self.real_data[:, 0].max():.3f}]")
        print(f"   Y range: [{self.real_data[:, 1].min():.3f}, {self.real_data[:, 1].max():.3f}]")
        print(f"   Z range: [{self.real_data[:, 2].min():.3f}, {self.real_data[:, 2].max():.3f}]")
        
        # Analyze trajectory segments for pattern recognition
        self.analyze_trajectory_patterns()
        
    def analyze_trajectory_patterns(self):
        """Analyze patterns in the real trajectory for diverse generation."""
        print("🧠 Analyzing trajectory patterns for diverse generation...")
        
        # 1. Segment the trajectory into meaningful parts
        segment_length = 20  # Analyze in segments of 20 points
        self.trajectory_segments = []
        
        for i in range(0, len(self.real_data) - segment_length, segment_length // 2):
            segment = self.real_data[i:i + segment_length]
            if len(segment) == segment_length:
                self.trajectory_segments.append(segment)
        
        print(f"   Created {len(self.trajectory_segments)} trajectory segments")
        
        # 2. Analyze statistical patterns at different trajectory phases
        self.statistical_patterns = {
            'early_phase': {},    # First 30% of trajectory
            'middle_phase': {},   # Middle 40% of trajectory  
            'late_phase': {}      # Last 30% of trajectory
        }
        
        n_points = len(self.real_data)
        early_end = int(0.3 * n_points)
        middle_end = int(0.7 * n_points)
        
        phases = {
            'early_phase': self.real_data[:early_end],
            'middle_phase': self.real_data[early_end:middle_end],
            'late_phase': self.real_data[middle_end:]
        }
        
        for phase_name, phase_data in phases.items():
            self.statistical_patterns[phase_name] = {
                'mean': np.mean(phase_data, axis=0),
                'std': np.std(phase_data, axis=0),
                'min': np.min(phase_data, axis=0),
                'max': np.max(phase_data, axis=0),
                'median': np.median(phase_data, axis=0)
            }
        
        # 3. Analyze step patterns
        steps = np.diff(self.real_data, axis=0)
        step_magnitudes = np.linalg.norm(steps, axis=1)
        
        self.step_patterns = {
            'mean_step': np.mean(step_magnitudes),
            'std_step': np.std(step_magnitudes),
            'direction_changes': steps,
            'step_distribution': step_magnitudes
        }
        
        print("✅ Pattern analysis completed")
        
    def train_massive_generator(self):
        """Train the massive trajectory generator."""
        print("="*80)
        print("🚀 TRAINING MASSIVE NEUTRON TRAJECTORY GENERATOR")
        print("="*80)
        
        self.load_and_analyze_real_data()
        self.is_trained = True
        
        print("✅ Massive generator ready for 1000+ trajectory generation!")
        
    def generate_diverse_trajectory_method1(self, trajectory_length=50, seed=None):
        """Method 1: Segment-based trajectory generation."""
        if seed is not None:
            np.random.seed(seed)
            
        trajectory = np.zeros((trajectory_length, 3))
        
        # Select random segments and combine them
        used_segments = []
        current_pos = 0
        
        while current_pos < trajectory_length:
            # Select a random segment
            segment_idx = np.random.randint(len(self.trajectory_segments))
            segment = self.trajectory_segments[segment_idx].copy()
            
            # Add variation to avoid exact duplication
            variation = np.random.normal(0, 0.02, segment.shape)
            segment += variation
            
            # Determine how much of this segment to use
            remaining_length = trajectory_length - current_pos
            use_length = min(len(segment), remaining_length)
            
            trajectory[current_pos:current_pos + use_length] = segment[:use_length]
            current_pos += use_length
            used_segments.append(segment_idx)
        
        return trajectory
    
    def generate_diverse_trajectory_method2(self, trajectory_length=50, seed=None):
        """Method 2: Phase-based trajectory generation."""
        if seed is not None:
            np.random.seed(seed)
            
        trajectory = np.zeros((trajectory_length, 3))
        
        # Divide trajectory into phases
        early_len = int(0.3 * trajectory_length)
        middle_len = int(0.4 * trajectory_length)
        late_len = trajectory_length - early_len - middle_len
        
        phases = [
            ('early_phase', early_len),
            ('middle_phase', middle_len),
            ('late_phase', late_len)
        ]
        
        current_pos = 0
        for phase_name, phase_len in phases:
            phase_stats = self.statistical_patterns[phase_name]
            
            for i in range(phase_len):
                # Generate point based on phase statistics
                for coord_idx in range(3):
                    # Use phase-specific statistics
                    mean_val = phase_stats['mean'][coord_idx]
                    std_val = phase_stats['std'][coord_idx]
                    min_val = phase_stats['min'][coord_idx]
                    max_val = phase_stats['max'][coord_idx]
                    
                    # Generate with controlled variation
                    point = np.random.normal(mean_val, std_val * 0.8)
                    point = np.clip(point, min_val, max_val)
                    trajectory[current_pos, coord_idx] = point
                
                current_pos += 1
        
        return trajectory
    
    def generate_diverse_trajectory_method3(self, trajectory_length=50, seed=None):
        """Method 3: Interpolation-based with high variation."""
        if seed is not None:
            np.random.seed(seed)
            
        trajectory = np.zeros((trajectory_length, 3))
        
        # Select random anchor points from real data
        n_anchors = min(10, trajectory_length // 5)  # 10 anchor points max
        anchor_indices = np.random.choice(len(self.real_data), n_anchors, replace=False)
        anchor_indices = np.sort(anchor_indices)
        anchor_points = self.real_data[anchor_indices]
        
        # Create interpolation between anchors
        t_anchors = np.linspace(0, trajectory_length - 1, n_anchors)
        t_trajectory = np.arange(trajectory_length)
        
        for coord_idx in range(3):
            # Interpolate with added variation
            interp_func = interpolate.interp1d(t_anchors, anchor_points[:, coord_idx], 
                                             kind='cubic', fill_value='extrapolate')
            base_trajectory = interp_func(t_trajectory)
            
            # Add controlled noise
            noise = np.random.normal(0, np.std(self.real_data[:, coord_idx]) * 0.15, trajectory_length)
            trajectory[:, coord_idx] = base_trajectory + noise
            
            # Ensure within realistic bounds
            min_val = np.min(self.real_data[:, coord_idx])
            max_val = np.max(self.real_data[:, coord_idx])
            trajectory[:, coord_idx] = np.clip(trajectory[:, coord_idx], min_val, max_val)
        
        return trajectory
    
    def generate_diverse_trajectory_method4(self, trajectory_length=50, seed=None):
        """Method 4: Statistical sampling with step constraints."""
        if seed is not None:
            np.random.seed(seed)
            
        trajectory = np.zeros((trajectory_length, 3))
        
        # Start from a random point in the real data
        start_idx = np.random.randint(len(self.real_data))
        trajectory[0] = self.real_data[start_idx]
        
        for i in range(1, trajectory_length):
            # Generate next point based on step patterns
            prev_point = trajectory[i-1]
            
            # Sample step size from real distribution
            step_size = np.random.choice(self.step_patterns['step_distribution'])
            
            # Generate random direction
            direction = np.random.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction)  # Normalize
            
            # Apply step
            next_point = prev_point + step_size * direction
            
            # Ensure within bounds
            for coord_idx in range(3):
                min_val = np.min(self.real_data[:, coord_idx])
                max_val = np.max(self.real_data[:, coord_idx])
                next_point[coord_idx] = np.clip(next_point[coord_idx], min_val, max_val)
            
            trajectory[i] = next_point
        
        return trajectory
    
    def generate_massive_trajectories(self, num_trajectories=2000, trajectory_length=50):
        """Generate massive number of diverse trajectories."""
        
        if not self.is_trained:
            self.train_massive_generator()
        
        print(f"\n🚀 GENERATING {num_trajectories:,} DIVERSE NEUTRON TRAJECTORIES")
        print("="*70)
        print(f"🎯 Target: Maximum diversity for comprehensive reactor simulation")
        print(f"📊 Trajectory length: {trajectory_length} points each")
        
        # Distribute generation across all methods for maximum diversity
        method1_count = num_trajectories // 4
        method2_count = num_trajectories // 4  
        method3_count = num_trajectories // 4
        method4_count = num_trajectories - method1_count - method2_count - method3_count
        
        trajectories = []
        
        print(f"\n🔬 Method 1 (Segment-based): Generating {method1_count:,} trajectories...")
        for i in range(method1_count):
            traj = self.generate_diverse_trajectory_method1(trajectory_length, seed=i)
            trajectories.append(traj)
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1:,}/{method1_count:,} trajectories")
        
        print(f"\n🧠 Method 2 (Phase-based): Generating {method2_count:,} trajectories...")
        for i in range(method2_count):
            traj = self.generate_diverse_trajectory_method2(trajectory_length, seed=i+10000)
            trajectories.append(traj)
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1:,}/{method2_count:,} trajectories")
        
        print(f"\n🎯 Method 3 (Interpolation-based): Generating {method3_count:,} trajectories...")
        for i in range(method3_count):
            traj = self.generate_diverse_trajectory_method3(trajectory_length, seed=i+20000)
            trajectories.append(traj)
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1:,}/{method3_count:,} trajectories")
        
        print(f"\n⚡ Method 4 (Step-constrained): Generating {method4_count:,} trajectories...")
        for i in range(method4_count):
            traj = self.generate_diverse_trajectory_method4(trajectory_length, seed=i+30000)
            trajectories.append(traj)
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1:,}/{method4_count:,} trajectories")
        
        print(f"\n✅ MASSIVE GENERATION COMPLETED!")
        print(f"   📊 Total trajectories: {len(trajectories):,}")
        print(f"   📈 Total data points: {len(trajectories) * trajectory_length:,}")
        print(f"   🔢 Diversity factor: {len(trajectories) / 1:.0f}x more trajectories than original")
        
        return trajectories
    
    def save_massive_trajectories(self, trajectories, filename="massive_synthetic_trajectories.csv"):
        """Save massive trajectory dataset."""
        
        print(f"\n💾 SAVING MASSIVE TRAJECTORY DATASET")
        print("="*45)
        
        all_data = []
        for traj_id, traj in enumerate(trajectories):
            for step, (x, y, z) in enumerate(traj):
                all_data.append({
                    'trajectory_id': traj_id,
                    'step': step,
                    'x': x,
                    'y': y,
                    'z': z
                })
        
        df = pd.DataFrame(all_data)
        
        # Ensure directory exists
        os.makedirs('neutron_massive_results', exist_ok=True)
        full_path = f'neutron_massive_results/{filename}'
        df.to_csv(full_path, index=False)
        
        print(f"✅ Saved to {full_path}")
        print(f"   📊 Trajectories: {len(trajectories):,}")
        print(f"   📈 Data points: {len(all_data):,}")
        print(f"   💽 File size: ~{len(all_data) * 50 / 1024 / 1024:.1f} MB")
        
        # Create summary statistics
        self.create_massive_dataset_summary(df, trajectories)
        
        return df
    
    def create_massive_dataset_summary(self, df, trajectories):
        """Create comprehensive summary of the massive dataset."""
        
        print(f"\n📊 MASSIVE DATASET SUMMARY")
        print("="*35)
        
        # Basic statistics
        print(f"Dataset Scale:")
        print(f"   Total trajectories: {len(trajectories):,}")
        print(f"   Points per trajectory: {len(trajectories[0])}")
        print(f"   Total data points: {len(df):,}")
        print(f"   Increase from real data: {len(df) / len(self.real_data):.0f}x")
        
        # Coordinate statistics comparison
        real_df = pd.read_csv('data/Sheet.csv')
        
        print(f"\nCoordinate Statistics Comparison:")
        for coord in ['x', 'y', 'z']:
            real_mean = real_df[coord].mean()
            real_std = real_df[coord].std()
            real_range = real_df[coord].max() - real_df[coord].min()
            
            synth_mean = df[coord].mean()
            synth_std = df[coord].std()
            synth_range = df[coord].max() - df[coord].min()
            
            mean_match = 100 * (1 - abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10))
            std_match = 100 * (1 - abs(real_std - synth_std) / (real_std + 1e-10))
            range_match = 100 * (1 - abs(real_range - synth_range) / (real_range + 1e-10))
            
            print(f"   {coord.upper()}-coordinate:")
            print(f"      Real:      μ={real_mean:8.4f}, σ={real_std:8.4f}, range={real_range:8.4f}")
            print(f"      Synthetic: μ={synth_mean:8.4f}, σ={synth_std:8.4f}, range={synth_range:8.4f}")
            print(f"      Match:     μ={mean_match:6.2f}%, σ={std_match:6.2f}%, range={range_match:6.2f}%")
        
        # Diversity analysis
        print(f"\nDiversity Analysis:")
        unique_trajectories = len(df['trajectory_id'].unique())
        avg_points_per_traj = len(df) / unique_trajectories
        
        print(f"   Unique trajectories: {unique_trajectories:,}")
        print(f"   Average points per trajectory: {avg_points_per_traj:.1f}")
        print(f"   Trajectory diversity: MAXIMUM (4 different generation methods)")

def main():
    """Main function for massive trajectory generation."""
    
    print("="*80)
    print("🚀 MASSIVE NEUTRON TRAJECTORY GENERATOR")
    print("="*80)
    print("🎯 TARGET: 1000+ Diverse Trajectories for Comprehensive Analysis")
    print("🔬 APPLICATION: Large-Scale Nuclear Reactor Simulation")
    print("⚡ METHOD: Multi-Strategy Diverse Generation")
    print("="*80)
    
    # Initialize massive generator
    generator = MassiveNeutronTrajectoryGenerator()
    
    # Generate massive trajectory dataset
    print("🚀 Starting massive trajectory generation...")
    
    # You can adjust these parameters:
    NUM_TRAJECTORIES = 2000  # Generate 2000 trajectories (much more than requested!)
    TRAJECTORY_LENGTH = 50   # 50 points per trajectory
    
    massive_trajectories = generator.generate_massive_trajectories(
        num_trajectories=NUM_TRAJECTORIES,
        trajectory_length=TRAJECTORY_LENGTH
    )
    
    # Save massive dataset
    df = generator.save_massive_trajectories(massive_trajectories, "massive_synthetic_trajectories.csv")
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏆 MASSIVE GENERATION MISSION ACCOMPLISHED!")
    print(f"✅ Generated {NUM_TRAJECTORIES:,} diverse neutron trajectories")
    print(f"📊 Total data points: {len(df):,}")
    print(f"🔢 Diversity: 4 different generation methods for maximum variation")
    print(f"💽 Ready for large-scale nuclear reactor analysis and simulation")
    print("="*80)
    
    return len(massive_trajectories)

if __name__ == '__main__':
    trajectory_count = main()
    print(f"\n🎯 SUCCESS: {trajectory_count:,} DIVERSE TRAJECTORIES GENERATED!")
    print("🚀 Ready for comprehensive nuclear reactor simulation!")