import pandas as pd
import numpy as np

# Load real and generated data
real_df = pd.read_csv('Sheet.csv')
gen_df = pd.read_csv('generated_for_enrichment_cgan.csv')

# Filter out rows with NaN enrichment values in real data
real_df = real_df.dropna(subset=['Enrichment (%)'])

# Align both dataframes by enrichment value
merged = pd.merge(real_df, gen_df, on='Enrichment (%)', suffixes=('_real', '_gen'))

# Calculate errors
merged['Flux Error'] = np.abs(merged['Flux (n/cm²/s)'] - merged['Flux'])
merged['Burnup Error'] = np.abs(merged['Burnup (MWd/kgU)'] - merged['Burnup'])

# Select and rename columns for output
out_df = merged[['Enrichment (%)', 'Flux (n/cm²/s)', 'Flux', 'Burnup (MWd/kgU)', 'Burnup', 'Flux Error', 'Burnup Error']]
out_df.columns = ['Enrichment (%)', 'Real Flux', 'Generated Flux', 'Real Burnup', 'Generated Burnup', 'Flux Error', 'Burnup Error']

# Save to CSV
out_df.to_csv('code/point_to_point_comparison.csv', index=False)
print('Point-to-point comparison saved to code/point_to_point_comparison.csv') 