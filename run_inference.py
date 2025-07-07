#!/usr/bin/env python3
"""
FLUXGAN Inference Script for Binder
Run this to test the trained FLUXGAN model
"""

import os
import sys

def main():
    print("🔍 FLUXGAN Inference for Binder")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('./code/flux_burnup_dataset.csv'):
        print("❌ Error: Dataset not found!")
        print("Make sure flux_burnup_dataset.csv is in the code/ directory")
        return
    
    # Create plots directory
    os.makedirs('./code/plots', exist_ok=True)
    
    print("✅ Dataset found")
    print("✅ Directories created")
    print("✅ Starting inference...")
    print("=" * 50)
    
    # Import and run inference
    try:
        sys.path.append('./code')
        exec(open('./code/simple_test.py').read())
        print("✅ Inference completed successfully!")
        
        # Show results
        if os.path.exists('./code/plots/simple_test.png'):
            print("📊 Results saved to ./code/plots/simple_test.png")
            print("📄 Data saved to ./code/plots/simple_test_samples.csv")
        else:
            print("⚠️  No results generated")
            
    except Exception as e:
        print(f"❌ Inference error: {e}")
        print("Check the logs above for details")

if __name__ == "__main__":
    main() 