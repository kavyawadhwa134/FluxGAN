# Neutron Trajectory LSTM-GAN

This project adapts the LSTM-TrajGAN architecture for generating synthetic neutron trajectories from reactor pin cell data. The model learns to generate realistic 3D neutron paths based on your input data.

## Files Overview

### Core Files
- `neutron_model.py` - LSTM-GAN model architecture for 3D trajectories
- `neutron_train.py` - Training script for the neutron trajectory GAN
- `neutron_predict.py` - Generate synthetic trajectories using trained model
- `neutron_losses.py` - Custom loss functions for trajectory generation

### Data Processing
- `data/neutron_preprocessing.py` - Converts CSV data to model-ready format
- `data/Sheet.csv` - Your original neutron trajectory data (x,y,z coordinates)

### Generated Files
- `data/neutron_train_xyz.npy` - Processed trajectory sequences
- `data/neutron_train_mask.npy` - Masks indicating valid trajectory points
- `data/neutron_train_norm_params.npy` - Normalization parameters
- `neutron_params/` - Directory for saved model weights
- `neutron_results/` - Directory for generated plots and results

## Quick Start

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv neutron_trajgan_env
source neutron_trajgan_env/bin/activate

# Install dependencies
pip install pandas numpy tensorflow keras matplotlib
```

### 2. Data Preprocessing
```bash
# Process your neutron trajectory data
python data/neutron_preprocessing.py --sequence_length 50
```

### 3. Train the Model
```bash
# Train for 2000 epochs with batch size 16, save every 200 epochs
python neutron_train.py 2000 16 200

# Or use default parameters
python neutron_train.py
```

### 4. Generate Synthetic Trajectories
```bash
# Generate 20 trajectories using model from epoch 1800
python neutron_predict.py 1800 20

# Or use defaults
python neutron_predict.py
```

## Model Architecture

The neutron trajectory GAN consists of:

### Generator
- Takes real trajectory + noise as input
- LSTM layers (128 → 64 units) with L1 regularization
- Outputs 3D coordinates (x,y,z) with tanh activation
- Applies masking to handle variable-length trajectories

### Discriminator
- Takes trajectory sequences as input
- LSTM layers (128 → 64 units) with L1 regularization
- Binary classification (real vs synthetic)
- Uses sigmoid activation for final prediction

### Loss Functions
- **Adversarial Loss**: Standard GAN loss to fool discriminator
- **Reconstruction Loss**: MSE between real and synthetic trajectories
- **Smoothness Loss**: Penalizes large jumps between consecutive points

## Data Format

### Input Data
Your CSV should have three columns: `x,y,z` representing 3D coordinates of neutron positions.

### Processed Data
- Coordinates normalized to [-1, 1] range
- Sequences of fixed length (default: 50 time steps)
- Padding with zeros for shorter trajectories
- Masks to indicate valid vs padded positions

## Training Tips

1. **Sequence Length**: Adjust based on your trajectory lengths
   - Shorter sequences (20-50) for quick training
   - Longer sequences (100+) for more detailed trajectories

2. **Batch Size**: 
   - Smaller batches (8-16) for limited data
   - Larger batches (32-64) if you have many trajectories

3. **Epochs**: 
   - Start with 1000-2000 epochs
   - Monitor discriminator/generator loss balance
   - Stop if discriminator becomes too strong (accuracy >> 90%)

4. **Learning Rate**: 
   - Default: 0.0002 (Adam optimizer)
   - Reduce if training is unstable
   - Increase slightly if training is too slow

## Results

The model generates:
- 3D trajectory visualizations
- Comparison plots (real vs synthetic)
- CSV files with synthetic trajectory coordinates
- Statistical comparisons of coordinate distributions

## Customization

### Modify Architecture
Edit `neutron_model.py` to:
- Change LSTM layer sizes
- Add more layers
- Modify regularization strength

### Adjust Loss Functions
Edit `neutron_losses.py` to:
- Change loss weights
- Add physics-based constraints
- Include domain-specific penalties

### Data Augmentation
Modify `data/neutron_preprocessing.py` to:
- Add noise to trajectories
- Create rotated/scaled versions
- Include additional features

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Reduce learning rate
   - Increase regularization
   - Balance discriminator/generator updates

2. **Poor Quality Trajectories**
   - Increase reconstruction loss weight
   - Add more training data
   - Adjust sequence length

3. **Mode Collapse**
   - Reduce discriminator strength
   - Add noise to discriminator inputs
   - Use different GAN variants (WGAN, etc.)

### Memory Issues
- Reduce batch size
- Use shorter sequences
- Process data in chunks

## Next Steps

1. **Validation**: Compare synthetic trajectories with physics simulations
2. **Metrics**: Implement trajectory-specific evaluation metrics
3. **Conditioning**: Add conditional generation based on reactor parameters
4. **Physics**: Incorporate neutron physics constraints in loss functions