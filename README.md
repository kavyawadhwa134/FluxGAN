# FluxGAN v3: High-Fidelity Conditional GAN Workflow

This repository implements a high-fidelity conditional GAN (cGAN) for generating and predicting nuclear fuel cycle data (Flux and Burnup) conditioned on enrichment. The workflow includes training, inference, and point-to-point comparison with real OpenMC data.

## Requirements

Install dependencies using:
```
pip install -r requirements.txt
```

## Files
- `code/fluxgan_cgan.py`: cGAN training script
- `code/generate_for_enrichment_cgan.py`: Generate predictions for 100 enrichment values
- `code/compare_generated_vs_real.py`: Compare cGAN predictions to real OpenMC data
- `code/point_to_point_comparison.py`: Create a detailed comparison CSV
- `code/flux_burnup_dataset.csv`: Training dataset
- `Sheet.csv`: Real OpenMC data for comparison
- `generated_for_enrichment_cgan.csv`: cGAN predictions
- `code/point_to_point_comparison.csv`: Point-to-point comparison results

## How to Run

### 1. Train the cGAN
Train the conditional GAN using your dataset:
```bash
python code/fluxgan_cgan.py
```
This will train the cGAN and save checkpoints in `plots/checkpoint/`.

### 2. Generate predictions for enrichment values
After training, generate predictions for the 100 enrichment values:
```bash
python code/generate_for_enrichment_cgan.py
```
This will create `generated_for_enrichment_cgan.csv` with predictions for the 100 enrichment values.

### 3. Compare predictions to real data (Sheet.csv)
To compare the cGAN predictions to the real OpenMC data:
```bash
python code/compare_generated_vs_real.py
```
This prints MAE, RMSE, and R² for Flux and Burnup.

### 4. Create a point-to-point comparison CSV
For a detailed row-by-row comparison (including errors):
```bash
python code/point_to_point_comparison.py
```
This creates `code/point_to_point_comparison.csv` with detailed errors for each enrichment value.

## Notes
- Make sure `code/flux_burnup_dataset.csv` and `Sheet.csv` are present in the repository.
- All scripts assume you are running from the project root directory.
- For best results, train the cGAN for several thousand epochs.

## License
MIT 