#!/usr/bin/env python3
"""
Improved FluxGAN Training Runner
This script trains the improved physics-informed FluxGAN with better enrichment handling.
"""

import os
import sys
import subprocess
import time

def main():
    print("🚀 IMPROVED FLUXGAN TRAINING")
    print("=" * 50)
    print("This version fixes the enrichment accuracy issue by:")
    print("✅ Using actual data bounds for enrichment (1-89%)")
    print("✅ Adding separate enrichment-specific loss function")
    print("✅ Reducing physics weights for better stability")
    print("✅ Improving network architecture with dropout")
    print("✅ Better learning rate scheduling")
    print("=" * 50)
    
    # Check if the improved model file exists
    if not os.path.exists('code/improved_fluxgan.py'):
        print("❌ Error: code/improved_fluxgan.py not found!")
        return
    
    # Check if dataset exists
    if not os.path.exists('code/flux_burnup_dataset.csv'):
        print("❌ Error: code/flux_burnup_dataset.csv not found!")
        return
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    print("\n📊 Starting improved FluxGAN training...")
    print("⏱️  Expected training time: ~2-3 hours")
    print("💾 Checkpoints will be saved every 1000 epochs")
    print("📈 Loss logs will be saved to plots/loss_log_improved.csv")
    
    # Start training
    start_time = time.time()
    
    try:
        # Run the improved training script
        result = subprocess.run([
            sys.executable, 'code/improved_fluxgan.py'
        ], capture_output=True, text=True, check=True)
        
        print("\n✅ Training completed successfully!")
        print(f"⏱️  Total training time: {(time.time() - start_time) / 3600:.2f} hours")
        
        # Show final message
        print("\n🎯 NEXT STEPS:")
        print("1. Generate samples: python code/generate_improved_samples.py")
        print("2. Analyze results: python code/analyze_improved_results.py")
        print("3. Compare with previous model")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return

if __name__ == "__main__":
    main() 