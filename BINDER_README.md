# FLUXGAN on Binder

This repository is set up to run on Binder for easy access to the improved FLUXGAN model.

## 🚀 Quick Start on Binder

1. **Click the Binder badge** to launch the environment
2. **Upload your dataset** `flux_burnup_dataset.csv` to the `code/` directory
3. **Run training or inference**

## 📁 File Structure

```
FluxGAN/
├── code/
│   ├── flux_burnup_dataset.csv    # Your dataset (upload this)
│   ├── simple_improved_fluxgan.py # Main training script
│   ├── simple_test.py             # Inference/testing script
│   └── plots/                     # Results directory
├── run_training.py                # Binder training runner
├── run_inference.py               # Binder inference runner
└── requirements.txt               # Dependencies
```

## 🔧 How to Run

### **Option 1: Training (Train the Model)**
```bash
python run_training.py
```
**What it does:**
- Trains the improved FLUXGAN for 5000 epochs
- Saves checkpoints every 500 epochs
- Generates loss plots and logs
- Takes ~30-60 minutes on Binder

### **Option 2: Inference (Test the Model)**
```bash
python run_inference.py
```
**What it does:**
- Tests the current model (trained or untrained)
- Generates sample data
- Creates comparison plots
- Takes ~2-5 minutes

### **Option 3: Direct Scripts**
```bash
# Training
python code/simple_improved_fluxgan.py

# Testing
python code/simple_test.py
```

## 📊 Expected Results

### **Training Output:**
```
Epoch [0/5000] | D: 0.6881 | G: 0.6997 | F: 0.0000
Epoch [10/5000] | D: 0.1946 | G: 4.7154 | F: 0.0000
...
Epoch [500/5000] | D: 0.1637 | G: 9.2085 | F: 0.0000
[Checkpoint] Saved at epoch 500
```

### **Inference Output:**
```
Generated samples:
  Enrichment: 47.78 - 52.05
  Flux: 4.76 - 5.21
  Burnup: 4.78e-08 - 5.19e-08
```

## 📈 Model Improvements

- ✅ **Enhanced Architecture**: Separate output heads for each feature
- ✅ **Better Training**: AdamW optimizer with learning rate scheduling
- ✅ **Improved Data**: RobustScaler for outlier handling
- ✅ **Stability**: Feature matching and gradient clipping
- ✅ **Monitoring**: Comprehensive logging and checkpointing

## 🎯 Key Features

1. **Training Script**: `simple_improved_fluxgan.py`
2. **Inference Script**: `simple_test.py`
3. **Evaluation**: `evaluate_fluxgan.py`
4. **Clean Testing**: `clean_test.py`

## ⚠️ Important Notes

- **Dataset Required**: Upload `flux_burnup_dataset.csv` to `code/` directory
- **Training Time**: 30-60 minutes on Binder (CPU)
- **Memory**: ~2GB RAM required
- **Results**: Saved in `code/plots/` directory

## 🔍 Troubleshooting

**If training fails:**
- Check dataset is uploaded correctly
- Ensure enough memory (restart kernel if needed)
- Check logs for specific errors

**If inference fails:**
- Run training first to generate checkpoints
- Check dataset format matches expected structure

## 📞 Support

For issues or questions:
1. Check the error logs
2. Verify dataset format
3. Restart the Binder environment if needed

---

**Happy Training! 🚀** 