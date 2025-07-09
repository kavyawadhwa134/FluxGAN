#!/usr/bin/env python3
"""
Script to run working FluxGAN training
"""

import os
import sys
import subprocess
import time

def main():
    print("=" * 60)
    print("Starting Working FluxGAN Training")
    print("=" * 60)
    print("This version focuses on:")
    print("- Working physics constraints (no problematic neutronics)")
    print("- Temperature consistency")
    print("- Thermal-hydraulics")
    print("- Fuel performance")
    print("- Burnup-flux correlation")
    print("=" * 60)
    
    # Check if training script exists
    training_script = './code/working_fluxgan.py'
    if not os.path.exists(training_script):
        print(f"Training script not found: {training_script}")
        return
    
    print(f"Running: {training_script}")
    print("Press Ctrl+C to stop training early")
    print("-" * 60)
    
    try:
        start_time = time.time()
        
        # Run the training script
        result = subprocess.run([sys.executable, training_script], 
                              capture_output=False, 
                              text=True)
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        if result.returncode == 0:
            print("-" * 60)
            print("Working FluxGAN training completed successfully!")
            print(f"Total training time: {training_duration/3600:.2f} hours")
            print("Checkpoints saved in: ./plots/checkpoint_working/")
            print("Loss log saved in: ./plots/loss_log_working.csv")
        else:
            print(f"Training failed with return code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Partial results may be available in checkpoint files")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")

if __name__ == "__main__":
    main() 