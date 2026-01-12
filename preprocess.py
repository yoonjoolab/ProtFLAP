#!/usr/bin/env python3
"""
Tinker Minimization + Energy Analysis Pipeline

- Minimizes PDB(s) given via -i/--input (produces *_min.pdb in-place or in -o folder).
- Generates atom-level CSV: <base>_min_energy.csv
- Generates per-residue averaged CSV: <base>_min.csv
- Cleans up intermediate files (.xyz, .txt, .seq).
"""

import subprocess
import os
import glob
import argparse
from pathlib import Path
import csv
import pandas as pd

# ===============================
# Configuration
# ===============================
tinker_dir = "/path/to/Tinker"
force = "/path/to/force.key" 
min_grid = "0.01"

pdb2xyz = tinker_dir/"pdbxyz"
xyz2pdb = tinker_dir/"xyzpdb"
minimize = tinker_dir/"minimize"
analyze = tinker_dir/"analyze"

# Columns for per-residue averaging
energy_columns = [
    'EB', 'EA', 'EBA', 'EUB', 'EAA', 'EOPB', 'EOPD', 'EID',
    'EIT', 'ET', 'EPT', 'EBT', 'EAT', 'ETT', 'EV', 'ER',
    'EDSP', 'EC', 'ECD', 'ED', 'EM', 'EP', 'ECT', 'ERXF',
    'ES', 'ELF', 'EG', 'EX'
]

# ===============================
# Argument parsing
# ===============================
parser = argparse.ArgumentParser(description="Tinker PDB Minimization + Energy Analysis")
parser.add_argument("-i", "--input", nargs="+", required=True, help="Input PDB file(s) or wildcards")
parser.add_argument("-o", "--output", default=None, help="Optional output directory for minimized PDBs")
args = parser.parse_args()

# Expand input wildcards
pdb_files = []
for item in args.input:
    pdb_files += glob.glob(item)

if not pdb_files:
    print("❌ No PDB files found for input:", args.input)
    exit(1)

# Output folder
if args.output:
    outdir = Path(args.output)
    outdir.mkdir(exist_ok=True)
else:
    outdir = None  # in-place

# ===============================
# MAIN LOOP: Minimize
# ===============================
for pdb_path in pdb_files:
    pdb_path = Path(pdb_path)
    if not pdb_path.exists() or pdb_path.suffix.lower() != ".pdb":
        print(f"⚠️ Skipping invalid file: {pdb_path}")
        continue

    base = pdb_path.stem
    work_dir = outdir if outdir else pdb_path.parent

    print("="*60)
    print(f"🔹 Minimizing: {pdb_path}")
    print(f"   Output dir: {work_dir}")
    print("="*60)

    # Step 1: Add hydrogens
    subprocess.run([str(pdb2xyz), str(pdb_path), "-k", str(force), "ALL", "A", "ALL"], check=True)
    subprocess.run([str(xyz2pdb), f"{base}.xyz", "-k", str(force)], check=True)
    hydro_pdb = work_dir / f"{base}_hydro.pdb"
    Path(f"{base}.pdb_2").rename(hydro_pdb)
    subprocess.run([str(pdb2xyz), str(hydro_pdb), "-k", str(force), "ALL", "A", "ALL"], check=True)

    # Step 2: Minimize
    hydro_xyz = work_dir / f"{base}_hydro.xyz"
    subprocess.run([str(minimize), str(hydro_xyz), "-k", str(force), min_grid], check=True)
    subprocess.run([str(xyz2pdb), f"{hydro_xyz}_2", "-k", str(force)], check=True)

    # Rename minimized PDB
    min_pdb = work_dir / f"{base}_min.pdb"
    Path(f"{base}_hydro.pdb_2").rename(min_pdb)

    # Cleanup intermediate files
    for f in [
        Path(f"{base}.xyz"),
        Path(f"{base}.seq"),
        hydro_pdb,
        Path(f"{base}_hydro.seq"),
        hydro_xyz,
        Path(f"{base}_hydro.xyz_2"),
        Path(f"{base}_hydro.pdb_2")
    ]:
        if f.exists():
            f.unlink()

    print(f"✅ Minimized PDB created: {min_pdb}")

print("🎉 All PDBs minimized successfully.")

# ===============================
# Energy Analysis on minimized PDBs
# ===============================
min_pdb_files = glob.glob(str(outdir / "*_min.pdb") if outdir else "*_min.pdb")
if not min_pdb_files:
    print("❌ No *_min.pdb files found for analysis")
    exit(1)

for pdb_file in min_pdb_files:
    # Skip small files
    if os.path.getsize(pdb_file) < 1000:
        print(f"Skipping small PDB file: {pdb_file}")
        continue

    pdb_base = Path(pdb_file).stem  # e.g., FlaA_min
    output_csv = f"{pdb_base}_energy.csv"     # Atom-level CSV
    averaged_csv = f"{pdb_base}.csv"          # Per-residue CSV

    print(f"\n🔹 Analyzing {pdb_file}")
    print(f"Atom-level CSV: {output_csv}")
    print(f"Averaged CSV: {averaged_csv}")

    # Step 1: Convert minimized PDB -> XYZ
    try:
        subprocess.run([pdb2xyz, pdb_file, "-k", force], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Failed to run pdbxyz for {pdb_file}")
        continue

    xyz_file = f"{pdb_base}.xyz"
    txt_file = f"{pdb_base}.txt"

    # Step 2: Run Tinker analyze
    try:
        with open(txt_file, "w") as f:
            subprocess.run([analyze, xyz_file, "-k", force, "A"], stdout=f, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Failed to run analyze for {pdb_file}")
        continue

    if not os.path.isfile(txt_file) or os.path.getsize(txt_file) == 0:
        print(f"❌ analyze output missing for {pdb_file}")
        continue

    # Step 3: Clean analyze TXT output
    with open(txt_file, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    skip = True
    for line in lines:
        if "Potential Energy Breakdown over Atoms :" in line:
            skip = False
            continue
        if not skip and line.strip():
            cleaned_lines.append(line.strip())

    # Step 4: TXT -> Atom-level CSV
    residues = {}
    with open(pdb_file, "r") as pdb_f:
        for line in pdb_f:
            if line.startswith(("ATOM", "HETATM")):
                atom_num = line[6:11].strip()
                res_num = line[22:26].strip()
                residues[atom_num] = res_num

    atom_data = []
    with open(output_csv, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        for line in cleaned_lines:
            values = line.split()
            if len(values) >= 7:
                if atom_data:
                    atom_num = atom_data[0]
                    writer.writerow([residues.get(atom_num, "Residue")] + atom_data)
                atom_data = [values[0]] + values[1:]
            else:
                atom_data.extend(values)

            if len(atom_data) == 29:
                atom_num = atom_data[0]
                writer.writerow([residues.get(atom_num, "Residue")] + atom_data)
                atom_data = []

        if atom_data:
            atom_num = atom_data[0]
            writer.writerow([residues.get(atom_num, "Residue")] + atom_data)

    # Step 5: Average per-residue (safe)
    df = pd.read_csv(output_csv)
    df['Residue'] = df['Residue'].ffill()
    available_columns = [c for c in energy_columns if c in df.columns]

    if not available_columns:
        print(f"⚠️ No matching energy columns found for {pdb_file}, skipping averaging.")
        continue

    averaged_df = df.groupby('Residue')[available_columns].mean().reset_index()
    averaged_df.to_csv(averaged_csv, index=False, float_format='%.4f')

    # Step 6: Cleanup intermediate files
    for f in [xyz_file, txt_file, f"{pdb_base}.seq"]:
        if os.path.exists(f):
            os.remove(f)

    print(f"✅ Finished processing {pdb_file}")

print("🎉 All PDBs minimized and analyzed successfully.")

