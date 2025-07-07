#!/usr/bin/env python3
"""
FLUXGAN Training Script for Binder
Run this to train the improved FLUXGAN model
"""

import os
import sys

def main():
    print("🚀 FLUXGAN Training for Binder")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('./code/flux_burnup_dataset.csv'):
        print("❌ Error: Dataset not found!")
        print("Make sure flux_burnup_dataset.csv is in the code/ directory")
        return
    
    # Create plots directory
    os.makedirs('./code/plots/checkpoint', exist_ok=True)
    
    print("✅ Dataset found")
    print("✅ Directories created")
    print("✅ Starting training...")
    print("=" * 50)
    
    # Import and run training
    try:
        sys.path.append('./code')
        exec(open('./code/simple_improved_fluxgan.py').read())
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Check the logs above for details")

if __name__ == "__main__":
    main() 