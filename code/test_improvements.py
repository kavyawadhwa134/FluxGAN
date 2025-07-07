import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import os

# Import the improved model
from simple_improved_fluxgan import ImprovedGenerator, ImprovedDiscriminator

def test_improved_model():
    """Test the improved model and compare with original data"""
    
    print("Testing Improved FLUXGAN Model")
    print("=" * 50)
    
    # Load original data
    data = pd.read_csv('./flux_burnup_dataset.csv')
    X_original = data[['Enrichment (%)', 'Flux', 'Burnup']].values
    
    print(f"Original data shape: {X_original.shape}")
    print(f"Original data ranges:")
    print(f"  Enrichment: {X_original[:, 0].min():.2f} - {X_original[:, 0].max():.2f}")
    print(f"  Flux: {X_original[:, 1].min():.2f} - {X_original[:, 1].max():.2f}")
    print(f"  Burnup: {X_original[:, 2].min():.2e} - {X_original[:, 2].max():.2e}")
    
    # Initialize improved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = ImprovedGenerator(noise_dim=128).to(device)
    
    # Generate samples with improved model
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1000, 128, device=device)
        conditions = torch.randn(1000, 3, device=device)
        fake_samples = generator(z, conditions).cpu().numpy()
    
    print(f"\nGenerated samples shape: {fake_samples.shape}")
    print(f"Generated data ranges:")
    print(f"  Enrichment: {fake_samples[:, 0].min():.2f} - {fake_samples[:, 0].max():.2f}")
    print(f"  Flux: {fake_samples[:, 1].min():.2f} - {fake_samples[:, 1].max():.2f}")
    print(f"  Burnup: {fake_samples[:, 2].min():.2e} - {fake_samples[:, 2].max():.2e}")
    
    # Compare statistics
    print(f"\nStatistical Comparison:")
    print(f"{'Feature':<15} {'Original Mean':<15} {'Generated Mean':<15} {'Mean Diff %':<12}")
    print("-" * 60)
    
    features = ['Enrichment (%)', 'Flux', 'Burnup']
    for i, feature in enumerate(features):
        orig_mean = X_original[:, i].mean()
        gen_mean = fake_samples[:, i].mean()
        mean_diff_pct = abs(orig_mean - gen_mean) / orig_mean * 100
        
        print(f"{feature:<15} {orig_mean:<15.4f} {gen_mean:<15.4f} {mean_diff_pct:<12.2f}%")
    
    print(f"\n{'Feature':<15} {'Original Std':<15} {'Generated Std':<15} {'Std Diff %':<12}")
    print("-" * 60)
    
    for i, feature in enumerate(features):
        orig_std = X_original[:, i].std()
        gen_std = fake_samples[:, i].std()
        std_diff_pct = abs(orig_std - gen_std) / orig_std * 100
        
        print(f"{feature:<15} {orig_std:<15.4f} {gen_std:<15.4f} {std_diff_pct:<12.2f}%")
    
    # Create simple visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, feature in enumerate(features):
        axes[i].hist(X_original[:, i], bins=30, alpha=0.7, label='Original', density=True)
        axes[i].hist(fake_samples[:, i], bins=30, alpha=0.7, label='Generated', density=True)
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./plots/improvement_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save generated samples
    results_df = pd.DataFrame(fake_samples, columns=features)
    results_df.to_csv('./plots/test_generated_samples.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  - ./plots/improvement_comparison.png")
    print(f"  - ./plots/test_generated_samples.csv")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    
    # Check if ranges are reasonable
    enrichment_ok = 0 <= fake_samples[:, 0].min() <= fake_samples[:, 0].max() <= 100
    flux_ok = 0 <= fake_samples[:, 1].min() <= fake_samples[:, 1].max() <= 15
    burnup_ok = 0 <= fake_samples[:, 2].min() <= fake_samples[:, 2].max() <= 1e-6
    
    print(f"  Enrichment range (0-100): {'✓' if enrichment_ok else '✗'}")
    print(f"  Flux range (0-15): {'✓' if flux_ok else '✗'}")
    print(f"  Burnup range (0-1e-6): {'✓' if burnup_ok else '✗'}")
    
    # Check for mode collapse
    enrichment_unique = len(np.unique(fake_samples[:, 0].round(2)))
    flux_unique = len(np.unique(fake_samples[:, 1].round(2)))
    burnup_unique = len(np.unique(fake_samples[:, 2].round(8)))
    
    print(f"  Enrichment diversity: {enrichment_unique}/1000 unique values")
    print(f"  Flux diversity: {flux_unique}/1000 unique values")
    print(f"  Burnup diversity: {burnup_unique}/1000 unique values")
    
    if enrichment_unique > 100 and flux_unique > 100 and burnup_unique > 100:
        print("  ✓ Good diversity - no mode collapse detected")
    else:
        print("  ⚠ Potential mode collapse - low diversity")

def demonstrate_improvements():
    """Demonstrate the key improvements made"""
    
    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS DEMONSTRATION")
    print("=" * 60)
    
    print("\n1. SEPARATE OUTPUT HEADS")
    print("   - Each feature (enrichment, flux, burnup) has its own output head")
    print("   - Proper scaling for each feature's range")
    print("   - Better control over individual feature generation")
    
    print("\n2. IMPROVED DATA PREPROCESSING")
    print("   - RobustScaler instead of MinMaxScaler")
    print("   - Better handling of outliers in burnup data")
    print("   - More stable training with extreme values")
    
    print("\n3. FEATURE MATCHING LOSS")
    print("   - Discriminator outputs both validity and features")
    print("   - Generator trained to match feature distributions")
    print("   - Better preservation of data relationships")
    
    print("\n4. ENHANCED TRAINING STRATEGY")
    print("   - AdamW optimizer with weight decay")
    print("   - Cosine annealing learning rate scheduling")
    print("   - Alternating training frequency")
    print("   - Better gradient flow and stability")
    
    print("\n5. ARCHITECTURE IMPROVEMENTS")
    print("   - Deeper network with layer normalization")
    print("   - Dropout for regularization")
    print("   - Separate noise and condition processing")
    print("   - Residual connections for better gradient flow")

if __name__ == "__main__":
    test_improved_model()
    demonstrate_improvements() 