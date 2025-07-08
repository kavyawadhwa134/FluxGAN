# FluxGAN-Multiphysics

FluxGAN-Multiphysics is a Generative Adversarial Network (GAN) designed for the **generation and prediction of multiphysics data in nuclear reactor analysis**. It can generate realistic samples of key reactor parameters, including neutron flux, fuel burnup, and temperature fields, enabling advanced data-driven studies and synthetic dataset creation for nuclear engineering applications.

---

## Domain & Application

**FluxGAN-Multiphysics** is tailored for the nuclear engineering domain, specifically for:
- **Reactor physics and thermal-hydraulics**
- Generation of synthetic multiphysics datasets
- Surrogate modeling for reactor core analysis
- Data augmentation for machine learning in nuclear science

**Predicted/Generated Quantities:**
- Neutron Flux (n/cm²/s)
- Burnup (MWd/kgU)
- Fuel Centerline Temperature (K)
- Clad Surface Temperature (K)
- Coolant Outlet Temperature (K)

---

## Model Results

| Quantity                    | MAE         | RMSE        | R²        | Accuracy (%) |
|-----------------------------|-------------|-------------|-----------|--------------|
| Flux (n/cm²/s)              | 0.0492      | 0.1109      | 0.9838    | 99.34%       |
| Burnup (MWd/kgU)            | 4.71e-10    | 8.34e-10    | 0.9565    | 98.83%       |
| Fuel Centerline Temp (K)    | 0.522       | 0.902       | 0.9566    | 99.92%       |
| Clad Surface Temp (K)       | 0.494       | 0.853       | 0.9567    | 99.92%       |
| Coolant Outlet Temp (K)     | 0.0175      | 0.0303      | 0.9568    | 99.997%      |

- **All R² values > 0.95**: The model explains over 95% of the variance for each quantity.
- **Very low MAE/RMSE**: Predictions are highly accurate and reliable for scientific use.

---

## How to Use

### 1. **Setup**
- Clone the repository and navigate to the project directory.
- Create and activate a Python virtual environment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. **Prepare Data**
- Place your training data in `code/flux_burnup_dataset.csv` (see example format in the repo).

### 3. **Train the Model**
- To train the cGAN for multiphysics:
  ```bash
  python code/fluxgan_cgan.py
  ```
- Checkpoints will be saved in `plots/checkpoint/`.

### 4. **Generate Synthetic Samples**
- After training, generate new samples:
  ```bash
  python code/generate_for_enrichment_cgan.py
  ```
- Output: `generated_for_enrichment_cgan.csv`

### 5. **Evaluate Model Performance**
- To evaluate the GAN's accuracy and generate plots:
  ```bash
  python code/evaluate_fluxgan.py
  ```
- Results and plots will be saved in the `plots/` directory.

### 6. **Compare with Real Data**
- For detailed comparison and error analysis:
  ```bash
  python code/compare_generated_vs_real.py
  python code/point_to_point_comparison.py
  ```
- Outputs: printed metrics and CSVs for pointwise errors.

---

## Requirements
- Python 3.8+
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy

Install all requirements with:
```bash
pip install -r requirements.txt
```

---

## Citation
If you use FluxGAN-Multiphysics in your research, please cite this repository and acknowledge the authors.

---

## Contact
For questions, issues, or collaboration, please open an issue or contact [kavyawadhwa134](https://github.com/kavyawadhwa134). 