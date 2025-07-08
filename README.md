# FluxGAN v3: High-Fidelity Conditional GAN for Nuclear Fuel Cycle Data

> **FluxGAN v3** is a state-of-the-art, high-fidelity machine learning workflow for generating and predicting nuclear fuel cycle data—specifically neutron Flux and Burnup—conditioned on fuel Enrichment. Built on a Conditional Generative Adversarial Network (cGAN), this project enables:
>
> - **Accurate surrogate modeling** of complex OpenMC simulations
> - **Rapid data generation** for new enrichment scenarios
> - **Point-to-point comparison** with real simulation data
> - **Seamless integration** into research and engineering pipelines

---

<p align="center">
  <img src="https://img.shields.io/badge/GAN-Conditional-blue" alt="Conditional GAN">
  <img src="https://img.shields.io/badge/Accuracy-High-green" alt="High Accuracy">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License">
</p>

---

## ✨ Features
- **Conditional GAN (cGAN):** Predicts Flux and Burnup for any given enrichment value
- **High Fidelity:** R² > 0.95 for both Flux and Burnup (see comparison results)
- **Easy-to-Use Workflow:** Train, generate, and compare with just a few commands
- **Point-to-Point Analysis:** CSV output for detailed error analysis
- **Reproducible:** All scripts and data included

## 🚀 Quick Start

1. **Set up the environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Train the cGAN**
   ```bash
   python code/fluxgan_cgan.py
   ```
3. **Generate predictions**
   ```bash
   python code/generate_for_enrichment_cgan.py
   ```
4. **Compare with real data**
   ```bash
   python code/compare_generated_vs_real.py
   ```
5. **Create a point-to-point comparison CSV**
   ```bash
   python code/point_to_point_comparison.py
   ```

---

## 📂 Files
| File | Purpose |
|------|---------|
| `code/fluxgan_cgan.py` | cGAN training script |
| `code/generate_for_enrichment_cgan.py` | Generate predictions for 100 enrichment values |
| `code/compare_generated_vs_real.py` | Compare cGAN predictions to real OpenMC data |
| `code/point_to_point_comparison.py` | Create a detailed comparison CSV |
| `code/flux_burnup_dataset.csv` | Training dataset |
| `Sheet.csv` | Real OpenMC data for comparison |
| `generated_for_enrichment_cgan.csv` | cGAN predictions |
| `code/point_to_point_comparison.csv` | Point-to-point comparison results |

---

## 📈 Results (Sample)
| Metric   | Flux      | Burnup    |
|----------|-----------|-----------|
| MAE      | 0.0424    | 4.66e-10  |
| RMSE     | 0.1127    | 8.36e-10  |
| R²       | 0.9832    | 0.9564    |s

---

## 📝 Notes
- Make sure `code/flux_burnup_dataset.csv` and `Sheet.csv` are present in the repository.
- All scripts assume you are running from the project root directory.
- For best results, train the cGAN for several thousand epochs.

## 📜 License
MIT 