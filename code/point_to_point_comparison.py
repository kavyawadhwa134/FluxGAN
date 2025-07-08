import pandas as pd
import numpy as np

# Load real and generated data
real_df = pd.read_csv('Sheet.csv')
gen_df = pd.read_csv('generated_for_enrichment_cgan.csv')

# Filter out rows with NaN enrichment values in real data
real_df = real_df.dropna(subset=['Enrichment (%)'])

# Align both dataframes by enrichment value
merged = pd.merge(real_df, gen_df, on='Enrichment (%)', suffixes=('_real', '_gen'))

# Define shared_cols for comparison (excluding Enrichment (%))
shared_cols = [col for col in real_df.columns if col in gen_df.columns and col != 'Enrichment (%)']

# Calculate errors
# Remove any hardcoded field references below
# Only use shared_cols for all error calculations and outputs
for col in shared_cols:
    merged[f'{col} Error'] = np.abs(merged[f'{col}_real'] - merged[f'{col}_gen'])

# Select and rename columns for output
out_cols = ['Enrichment (%)']
for col in shared_cols:
    out_cols.extend([f'{col}_real', f'{col}_gen', f'{col} Error'])
out_df = merged[out_cols]

# Rename columns for clarity
rename_dict = {}
for col in shared_cols:
    rename_dict[f'{col}_real'] = f'Real {col}'
    rename_dict[f'{col}_gen'] = f'Generated {col}'
    rename_dict[f'{col} Error'] = f'{col} Error'
out_df = out_df.rename(columns=rename_dict)

# Save to CSV
out_df.to_csv('code/point_to_point_comparison.csv', index=False)
print('Point-to-point comparison saved to code/point_to_point_comparison.csv') 