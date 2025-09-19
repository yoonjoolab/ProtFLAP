#!/usr/bin/env python3
"""
Convert Tinker analyze.txt output + PDB into per-atom CSV.
Usage:
    python3 convert_to_csv.py minimized.pdb energy.txt output.csv
"""

import sys
import csv

if len(sys.argv) != 4:
    print("Usage: python3 convert_to_csv.py minimized.pdb energy.txt output.csv")
    sys.exit(1)

pdb_file = sys.argv[1]
txt_file = sys.argv[2]
csv_file = sys.argv[3]

# Map atom numbers to residue numbers from PDB
residues = {}
with open(pdb_file, "r") as pdb:
    for line in pdb:
        if line.startswith(("ATOM", "HETATM")):
            atom_number = line[6:11].strip()
            residue_number = line[22:26].strip()
            residues[atom_number] = residue_number

# Columns used in your original code
energy_columns = ['EB','EA','EBA','EUB','EAA','EOPB','EOPD','EID',
                  'EIT','ET','EPT','EBT','EAT','ETT','EV','ER',
                  'EDSP','EC','ECD','ED','EM','EP','ECT','ERXF',
                  'ES','ELF','EG','EX']

with open(txt_file, "r") as infile, open(csv_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Residue", "Atom"] + energy_columns)
    
    atom_data = []
    for line in infile:
        values = line.strip().split()
        if len(values) >= 7:
            if atom_data:
                atom_number = atom_data[0]
                res = residues.get(atom_number, "Residue")
                writer.writerow([res] + atom_data)
            atom_data = [values[0]] + values[1:]
        else:
            atom_data.extend(values)
        if len(atom_data) == 29:
            atom_number = atom_data[0]
            res = residues.get(atom_number, "Residue")
            writer.writerow([res] + atom_data)
            atom_data = []
    if atom_data:
        atom_number = atom_data[0]
        res = residues.get(atom_number, "Residue")
        writer.writerow([res] + atom_data)

print(f"✅ Converted {txt_file} → {csv_file}")

