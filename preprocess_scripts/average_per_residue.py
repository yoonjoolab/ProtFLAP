#!/usr/bin/env python3
"""
Average per-atom CSV into per-residue CSV.
Removes completely empty rows, renumbers residues sequentially,
and deletes the last row of the output.
Usage:
    python3 average_per_residue.py input.csv output_avg.csv
"""

import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python3 average_per_residue.py input.csv output_avg.csv")
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = sys.argv[2]

# Energy columns
energy_columns = ['EB','EA','EBA','EUB','EAA','EOPB','EOPD','EID',
                  'EIT','ET','EPT','EBT','EAT','ETT','EV','ER',
                  'EDSP','EC','ECD','ED','EM','EP','ECT','ERXF',
                  'ES','ELF','EG','EX']

df = pd.read_csv(input_csv)

# Convert energy columns to numeric, coerce errors to NaN
for col in energy_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where all columns (Residue + energy) are NaN
df = df.dropna(how='all')

# Drop rows where all energy columns are NaN
df = df.dropna(subset=energy_columns, how='all')

# Forward-fill Residue column
df['Residue'] = df['Residue'].ffill()

# Average per-residue
averaged_df = df.groupby('Residue')[energy_columns].mean().reset_index()

# Renumber residues sequentially 1,2,3...
averaged_df['Residue'] = pd.factorize(averaged_df['Residue'])[0] + 1

# Delete the last row
averaged_df = averaged_df.iloc[:-1]

# Save CSV
averaged_df.to_csv(output_csv, index=False, float_format='%.4f')
print(f"✅ Averaged per-residue CSV saved → {output_csv}")

