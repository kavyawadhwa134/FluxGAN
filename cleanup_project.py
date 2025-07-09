#!/usr/bin/env python3
"""
Project Cleanup Script for Physics-Informed FluxGAN
This script identifies and removes unnecessary files while keeping essential ones.
"""

import os
import shutil
import glob

def cleanup_project():
    print("🧹 FLUXGAN PROJECT CLEANUP")
    print("=" * 50)
    
    # Define essential files to KEEP
    essential_files = {
        # Core working model files
        'code/working_fluxgan.py': 'Main working FluxGAN model',
        'code/flux_burnup_dataset.csv': 'Original dataset',
        'code/error_accuracy_analysis.py': 'Error and accuracy analysis',
        'code/generate_and_analyze_samples.py': 'Sample generation and analysis',
        
        # Runner scripts
        'run_working_training.py': 'Main training runner',
        
        # Generated results
        'generated_samples_working.csv': 'Generated samples from working model',
        
        # Essential plots and checkpoints
        'plots/checkpoint_working/': 'Working model checkpoints',
        'plots/loss_log_working.csv': 'Working model training log',
        'plots/error_accuracy_analysis.png': 'Error analysis visualization',
        'plots/physics_relationships.png': 'Physics relationships plot',
        'plots/correlation_comparison.png': 'Correlation comparison',
        'plots/generated_vs_real_distributions.png': 'Distribution comparison',
        'plots/overall_performance_summary.csv': 'Performance summary',
        'plots/physics_accuracy_metrics.csv': 'Physics accuracy metrics',
        'plots/distribution_accuracy_metrics.csv': 'Distribution metrics',
        
        # Documentation
        'README.md': 'Main project documentation',
        'requirements.txt': 'Python dependencies',
        
        # Git
        '.git/': 'Git repository',
        '.gitignore': 'Git ignore file'
    }
    
    # Define files to REMOVE (old/experimental versions)
    files_to_remove = [
        # Old training scripts
        'run_basic_stable_training.py',
        'run_stable_physics_training.py', 
        'run_advanced_physics_training.py',
        'run_physics_informed_training.py',
        
        # Old model files
        'code/basic_stable_fluxgan.py',
        'code/stable_physics_informed_fluxgan.py',
        'code/advanced_physics_informed_fluxgan.py',
        'code/physics_informed_fluxgan.py',
        'code/fluxgan_cgan.py',
        'code/fluxgan_with_noise.py',
        
        # Old analysis files
        'code/generate_physics_informed_samples.py',
        'code/evaluate_fluxgan.py',
        'code/visualize_results.py',
        'code/point_to_point_comparison.py',
        'code/compare_generated_vs_real.py',
        'code/generate_for_enrichment.py',
        
        # Old generated samples
        'generated_for_enrichment_cgan.csv',
        'generated_for_enrichment.csv',
        'generated_samples.csv',
        'Sheet.csv',
        
        # Old plots and checkpoints
        'plots/checkpoint_advanced_physics/',
        'plots/checkpoint_basic_stable/',
        'plots/checkpoint_stable_physics/',
        'plots/checkpoint/',
        'plots/loss_log_basic_stable.csv',
        'plots/loss_log_stable_physics.csv',
        'plots/loss_log_advanced_physics.csv',
        'plots/loss_log_cgan.csv',
        'plots/loss_log.csv',
        
        # Old analysis plots
        'plots/mae_barplot.png',
        'plots/correlation_heatmaps.png',
        'plots/distribution_*.png',
        'plots/residual_*.png',
        'plots/parity_*.png',
        
        # Old data files
        'code/point_to_point_comparison.csv',
        
        # System files
        '.DS_Store',
        'code/.DS_Store',
        'plots/.DS_Store',
        
        # Cache files
        'code/__pycache__/',
        
        # Old documentation
        'PHYSICS_INFORMED_README.md',
        'add_physics_columns.py'
    ]
    
    # Define directories to clean
    dirs_to_clean = [
        'plots/checkpoint_advanced_physics',
        'plots/checkpoint_basic_stable', 
        'plots/checkpoint_stable_physics',
        'plots/checkpoint',
        'code/__pycache__'
    ]
    
    print("📋 ESSENTIAL FILES TO KEEP:")
    for file_path, description in essential_files.items():
        if os.path.exists(file_path):
            print(f"  ✅ {file_path} - {description}")
        else:
            print(f"  ⚠️  {file_path} - {description} (NOT FOUND)")
    
    print(f"\n🗑️  FILES TO REMOVE:")
    removed_count = 0
    
    # Remove individual files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  🗑️  Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Error removing {file_path}: {e}")
    
    # Remove directories
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"  🗑️  Removed directory: {dir_path}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Error removing directory {dir_path}: {e}")
    
    # Remove old plot files using glob patterns
    old_plot_patterns = [
        'plots/distribution_*.png',
        'plots/residual_*.png', 
        'plots/parity_*.png'
    ]
    
    for pattern in old_plot_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"  🗑️  Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Error removing {file_path}: {e}")
    
    print(f"\n✅ CLEANUP COMPLETE!")
    print(f"📊 Removed {removed_count} files/directories")
    
    # Show final project structure
    print(f"\n📁 FINAL PROJECT STRUCTURE:")
    show_project_structure()
    
    # Create a summary file
    create_cleanup_summary()

def show_project_structure():
    """Show the cleaned project structure"""
    structure = {
        'code/': [
            'working_fluxgan.py (Main model)',
            'flux_burnup_dataset.csv (Dataset)',
            'error_accuracy_analysis.py (Analysis)',
            'generate_and_analyze_samples.py (Generation)'
        ],
        'plots/': [
            'checkpoint_working/ (Model checkpoints)',
            'loss_log_working.csv (Training log)',
            'error_accuracy_analysis.png (Analysis plots)',
            'physics_relationships.png (Physics plots)',
            'correlation_comparison.png (Correlation plots)',
            'generated_vs_real_distributions.png (Distribution plots)',
            'overall_performance_summary.csv (Performance metrics)',
            'physics_accuracy_metrics.csv (Physics metrics)',
            'distribution_accuracy_metrics.csv (Distribution metrics)'
        ],
        'Root/': [
            'run_working_training.py (Training runner)',
            'generated_samples_working.csv (Generated data)',
            'README.md (Documentation)',
            'requirements.txt (Dependencies)',
            'cleanup_project.py (This script)'
        ]
    }
    
    for directory, files in structure.items():
        print(f"\n{directory}")
        for file in files:
            print(f"  📄 {file}")

def create_cleanup_summary():
    """Create a summary of the cleanup"""
    summary = """# FluxGAN Project Cleanup Summary

## What was kept:
- **working_fluxgan.py**: The main working physics-informed FluxGAN model
- **flux_burnup_dataset.csv**: Original nuclear reactor dataset
- **error_accuracy_analysis.py**: Comprehensive error and accuracy analysis
- **generate_and_analyze_samples.py**: Sample generation and analysis
- **run_working_training.py**: Main training runner script
- **checkpoint_working/**: Latest model checkpoints
- **All analysis results**: Performance metrics, plots, and generated samples

## What was removed:
- Old experimental model versions (basic_stable, stable_physics, advanced_physics)
- Outdated training scripts
- Old analysis files
- Redundant generated samples
- Old checkpoints and logs
- System files (.DS_Store, __pycache__)

## Project is now clean and contains only essential files for:
1. Training the physics-informed FluxGAN
2. Generating nuclear reactor data
3. Analyzing model performance
4. Evaluating physics accuracy

## To use the project:
1. Train: `python run_working_training.py`
2. Generate samples: `python code/generate_and_analyze_samples.py`
3. Analyze errors: `python code/error_accuracy_analysis.py`
"""
    
    with open('CLEANUP_SUMMARY.md', 'w') as f:
        f.write(summary)
    
    print(f"\n📄 Created CLEANUP_SUMMARY.md with project details")

if __name__ == "__main__":
    # Ask for confirmation
    print("⚠️  This will remove old/experimental files. Continue? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        cleanup_project()
    else:
        print("❌ Cleanup cancelled.") 