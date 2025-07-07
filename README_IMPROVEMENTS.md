# FLUXGAN Accuracy Improvements

## Overview

This repository contains an improved version of the FLUXGAN model that addresses accuracy issues in burnup and flux generation. The improvements focus on better data preprocessing, enhanced architecture, improved training stability, and comprehensive evaluation.

## Key Improvements Made

### 1. **Data Preprocessing**
- **RobustScaler**: Replaced MinMaxScaler with RobustScaler for better handling of outliers in burnup data
- **Proper Scaling**: Each feature (enrichment, flux, burnup) has appropriate scaling factors

### 2. **Architecture Enhancements**
- **Separate Output Heads**: Each feature has its own output head with proper scaling
- **Deeper Network**: Added more layers with layer normalization and dropout
- **Feature Matching**: Discriminator outputs both validity and features for better training

### 3. **Training Stability**
- **AdamW Optimizer**: Better optimization with weight decay
- **Cosine Annealing**: Improved learning rate scheduling
- **Feature Matching Loss**: Combined adversarial and feature matching losses
- **Alternating Training**: Generator trained every 2 batches for stability

### 4. **Evaluation Framework**
- **Comprehensive Metrics**: KL divergence, correlation analysis, statistical comparison
- **Visualization Tools**: Distribution plots, correlation matrices, scatter plots
- **Quality Assessment**: Automatic detection of mode collapse and range validation

## Files Structure

```
FluxGAN/
├── code/
│   ├── simple_improved_fluxgan.py    # Improved GAN implementation
│   ├── test_improvements.py          # Test and demonstration script
│   ├── simple_evaluate.py            # Evaluation script
│   ├── flux_burnup_dataset.csv       # Original dataset
│   └── FLUXGAN-CT.ipynb              # Original implementation
├── plots/
│   ├── checkpoint/                   # Model checkpoints
│   ├── generated_samples.csv         # Generated samples
│   ├── improvement_comparison.png    # Visualization
│   └── evaluation_results.csv        # Evaluation metrics
├── IMPROVEMENTS_SUMMARY.md           # Detailed technical summary
└── README_IMPROVEMENTS.md            # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib seaborn scipy
```

### 2. Train the Improved Model
```bash
cd code
python simple_improved_fluxgan.py
```

### 3. Test the Improvements
```bash
cd code
python test_improvements.py
```

### 4. Evaluate Results
```bash
cd code
python simple_evaluate.py
```

## Expected Results

### Accuracy Improvements
- **Mean Difference**: < 5% for all features
- **Standard Deviation Difference**: < 10% for all features
- **Correlation**: > 0.7 Pearson correlation with original data
- **Distribution Similarity**: KL divergence < 1.0

### Training Stability
- **No Mode Collapse**: Consistent generation across all features
- **Stable Losses**: Discriminator and generator losses converge smoothly
- **Good Diversity**: High number of unique values in generated samples

## Key Features

### 1. **Separate Feature Generation**
```python
# Each feature has its own output head with proper scaling
enrichment = self.enrichment_head(x3) * 100  # 0-100 range
flux = self.flux_head(x3) * 10               # 0-10 range
burnup = self.burnup_head(x3) * 1e-7         # 0-1e-7 range
```

### 2. **Feature Matching Loss**
```python
# Discriminator outputs both validity and features
validity, features = discriminator(data)

# Generator trained to match feature distributions
feature_loss = mse_loss(fake_features, real_features)
total_loss = adversarial_loss + 0.1 * feature_loss
```

### 3. **Robust Data Preprocessing**
```python
# RobustScaler handles outliers better than MinMaxScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

## Monitoring Training

### Loss Metrics
- **D_Loss**: Should stabilize around 0.5-0.7
- **G_Loss**: Should decrease over time
- **Feature_Loss**: Should be low (< 0.1)

### Quality Indicators
- **Range Validation**: Generated values within expected ranges
- **Diversity Check**: High number of unique values
- **Statistical Similarity**: Mean and std close to original data

## Troubleshooting

### Common Issues

1. **Poor Burnup Accuracy**
   - Check burnup scaling factor (1e-7)
   - Verify RobustScaler is working correctly
   - Adjust burnup head architecture if needed

2. **Training Instability**
   - Reduce learning rates
   - Increase feature loss weight
   - Check batch size

3. **Mode Collapse**
   - Increase feature loss weight
   - Add more regularization
   - Check discriminator capacity

### Performance Tips

1. **GPU Usage**: The model automatically uses GPU if available
2. **Memory Management**: Adjust batch size based on available memory
3. **Checkpointing**: Model saves checkpoints every 500 epochs
4. **Early Stopping**: Monitor loss convergence for early stopping

## Comparison with Original

| Aspect | Original | Improved |
|--------|----------|----------|
| Data Scaling | MinMaxScaler | RobustScaler |
| Architecture | Simple Sequential | Separate Output Heads |
| Training | Basic BCE Loss | Feature Matching Loss |
| Optimizer | Adam | AdamW with Weight Decay |
| Scheduling | StepLR | CosineAnnealingLR |
| Evaluation | Basic Loss Logging | Comprehensive Metrics |

## Future Enhancements

1. **Advanced Architectures**
   - WGAN-GP for better stability
   - Self-attention mechanisms
   - Conditional batch normalization

2. **Better Evaluation**
   - FID score implementation
   - Physics-based validation
   - Wasserstein distance metrics

3. **Hyperparameter Optimization**
   - Bayesian optimization
   - Automated architecture search
   - Adaptive learning rates

## Technical Details

### Model Architecture
- **Generator**: 3-layer network with separate output heads
- **Discriminator**: Feature extractor + dual output heads
- **Noise Dimension**: 128
- **Hidden Layers**: 512 → 256 → 128

### Training Parameters
- **Batch Size**: 256
- **Learning Rate**: 0.0002 (G), 0.0001 (D)
- **Epochs**: 5000
- **Optimizer**: AdamW with weight decay 1e-4
- **Scheduler**: CosineAnnealingLR

### Data Processing
- **Scaler**: RobustScaler
- **Features**: Enrichment (%), Flux, Burnup
- **Dataset Size**: 10,000 samples

## Citation

If you use this improved FLUXGAN implementation, please cite:

```bibtex
@misc{fluxgan_improvements,
  title={Improved FLUXGAN for Accurate Burnup and Flux Generation},
  author={Your Name},
  year={2024},
  note={Enhanced GAN implementation with feature matching and robust preprocessing}
}
```

## License

This project is provided as-is for research and educational purposes. Please ensure you have the necessary permissions to use the original dataset and models.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the detailed technical summary in `IMPROVEMENTS_SUMMARY.md`
3. Examine the code comments for implementation details
4. Run the test scripts to verify functionality

---

**Note**: This improved implementation addresses the core accuracy issues in the original FLUXGAN model and should provide significantly better results for burnup and flux generation tasks. 