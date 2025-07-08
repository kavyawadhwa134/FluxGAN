import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error

# Load data
real_df = pd.read_csv('Sheet.csv')
gen_df = pd.read_csv('generated_for_enrichment_cgan.csv')

# Merge on Enrichment (%)
merged = pd.merge(real_df, gen_df, on='Enrichment (%)', suffixes=('_real', '_gen'))

# Find shared columns (excluding Enrichment (%))
shared_cols = [col for col in real_df.columns if col in gen_df.columns and col != 'Enrichment (%)']

os.makedirs('plots', exist_ok=True)

# 1. Parity (scatter) plots
for col in shared_cols:
    plt.figure()
    plt.scatter(merged[f'{col}_real'], merged[f'{col}_gen'], alpha=0.6)
    minv, maxv = merged[[f'{col}_real', f'{col}_gen']].min().min(), merged[[f'{col}_real', f'{col}_gen']].max().max()
    plt.plot([minv, maxv], [minv, maxv], 'r--')
    plt.xlabel(f'Real {col}')
    plt.ylabel(f'Generated {col}')
    plt.title(f'Parity Plot: {col}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/parity_{col.replace("/","_per_").replace(" ","_")}.png')
    plt.show()

# 2. Residual plots
for col in shared_cols:
    plt.figure()
    plt.scatter(merged[f'{col}_real'], merged[f'{col}_gen'] - merged[f'{col}_real'], alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel(f'Real {col}')
    plt.ylabel('Residual (Generated - Real)')
    plt.title(f'Residual Plot: {col}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/residual_{col.replace("/","_per_").replace(" ","_")}.png')
    plt.show()

# 3. Distribution (KDE) plots
for col in shared_cols:
    plt.figure()
    sns.kdeplot(merged[f'{col}_real'], label='Real', fill=True)
    sns.kdeplot(merged[f'{col}_gen'], label='Generated', fill=True)
    plt.xlabel(col)
    plt.title(f'Distribution Comparison: {col}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/distribution_{col.replace("/","_per_").replace(" ","_")}.png')
    plt.show()

# 4. Correlation heatmaps
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.heatmap(merged[[f'{col}_real' for col in shared_cols]].corr(), annot=True, cmap='coolwarm')
plt.title('Real Data Correlation')
plt.subplot(1,2,2)
sns.heatmap(merged[[f'{col}_gen' for col in shared_cols]].corr(), annot=True, cmap='coolwarm')
plt.title('Generated Data Correlation')
plt.tight_layout()
plt.savefig('plots/correlation_heatmaps.png')
plt.show()

# 5. Error bar plots (MAE)
maes = [mean_absolute_error(merged[f'{col}_real'], merged[f'{col}_gen']) for col in shared_cols]
plt.figure()
plt.bar(shared_cols, maes)
plt.ylabel('MAE')
plt.title('Mean Absolute Error (MAE) for Each Quantity')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('plots/mae_barplot.png')
plt.show() 