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

### 1. Set up the environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the cGAN
```bash
python code/fluxgan_cgan.py
```
This will train the cGAN and save checkpoints in `plots/checkpoint/`.

### 3. Generate predictions for enrichment values
```bash
python code/generate_for_enrichment_cgan.py
```
This will create `generated_for_enrichment_cgan.csv` with predictions for the 100 enrichment values.

### 4. Compare predictions to real data
```bash
python code/compare_generated_vs_real.py
```
This prints MAE, RMSE, and R² for Flux and Burnup.

### 5. Create a point-to-point comparison CSV
```bash
python code/point_to_point_comparison.py
```
This creates `code/point_to_point_comparison.csv` with detailed row-by-row errors.

## Notes
- Make sure `code/flux_burnup_dataset.csv` and `Sheet.csv` are present in the repository.
- All scripts assume you are running from the project root directory.
- For best results, train the cGAN for several thousand epochs.

## License
MIT 