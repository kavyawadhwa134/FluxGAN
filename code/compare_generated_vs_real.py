import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load real and generated data
real_df = pd.read_csv('Sheet.csv')
gen_df = pd.read_csv('generated_for_enrichment_cgan.csv')

# Filter out rows with NaN enrichment values in real data
real_df = real_df.dropna(subset=['Enrichment (%)'])

# Align both dataframes by enrichment value
merged = pd.merge(real_df, gen_df, on='Enrichment (%)', suffixes=('_real', '_gen'))

# Find shared columns for comparison (excluding Enrichment (%))
shared_cols = [col for col in real_df.columns if col in gen_df.columns and col != 'Enrichment (%)']

def print_metrics(name, real, gen):
    mae = mean_absolute_error(real, gen)
    rmse = np.sqrt(mean_squared_error(real, gen))
    r2 = r2_score(real, gen)
    print(f"{name}:")
    print(f"  MAE:  {mae:.6g}")
    print(f"  RMSE: {rmse:.6g}")
    print(f"  R^2:  {r2:.6g}\n")

print("Comparison of cGAN-generated vs. real OpenMC data (Sheet.csv):\n")
for col in shared_cols:
    print_metrics(col, merged[f'{col}_real'].values, merged[f'{col}_gen'].values) 