#!/bin/bash
# ========================================
# Protein Preprocessing with Tinker
# Usage: ./preprocess.sh input.pdb
# Output: minimized.pdb, energy.csv, energy_avg.csv
# Only keeps original PDB, minimized PDB, and both CSVs
# ========================================

# === SETTINGS ===
# Update these paths to match your system
tinker="path/to/tinker/bin"               # Folder containing pdbxyz, xyzpdb, minimize, analyze
force="path/to/force.key"         # Force key file
param_file="path/to/amber99sb.prm"        # Parameter file
min_grid=0.01                             # Minimization grid step


# === PARSE INPUT ARGUMENTS ===
input=""
outdir=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            shift
            input="$1"
            ;;
        -o|--output)
            shift
            outdir="$1"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 -i input.pdb [-o output_folder]"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$input" ]; then
    echo "Error: Input PDB file is required."
    echo "Usage: $0 -i input.pdb [-o output_folder]"
    exit 1
fi

# Absolute paths
input="$(realpath "$input")"
input_dir="$(dirname "$input")"
base="$(basename "$input" .pdb)"

# Set output folder
if [ -z "$outdir" ]; then
    outdir="$input_dir"
else
    outdir="$(realpath "$outdir")"
fi
mkdir -p "$outdir"

echo "🔹 Processing PDB: $input"
echo "Output folder: $outdir"

# =========================
# Run Tinker in input folder
# =========================
pushd "$input_dir" > /dev/null

# STEP 1: Add hydrogens + prepare XYZ
"$pdb2xyz" "$base.pdb" -k "$force" "$param_file" ALL A ALL
"$xyz2pdb" "${base}.xyz" -k "$force" "$param_file"
mv "${base}.pdb_2" "$outdir/${base}_hydro.pdb"
"$pdb2xyz" "$outdir/${base}_hydro.pdb" -k "$force" "$param_file" ALL A ALL

# STEP 2: Minimize structure
"$minimize" "$outdir/${base}_hydro.xyz" -k "$force" "$param_file" "$min_grid"
"$xyz2pdb" "$outdir/${base}_hydro.xyz_2" -k "$force" "$param_file"
mv "$outdir/${base}_hydro.pdb_2" "$outdir/${base}_min.pdb"

# STEP 3: Energy breakdown
echo "amber99sb" | "$pdb2xyz" "$outdir/${base}_min.pdb" -k "$force" "$param_file"
"$analyze" "$outdir/${base}_min.xyz" "$param_file" A > "$outdir/${base}_energy.txt"

popd > /dev/null

# STEP 4: Convert results to CSV
python3 preprocess_scripts/convert_to_csv.py "$outdir/${base}_min.pdb" "$outdir/${base}_energy.txt" "$outdir/${base}_energy.csv"
python3 preprocess_scripts/average_per_residue.py "$outdir/${base}_energy.csv" "$outdir/${base}_min.csv"

# === STEP 5: Cleanup intermediate files ===
rm -f "$outdir/${base}.xyz" \
      "$outdir/${base}.seq" \
      "$outdir/${base}_hydro.pdb" \
      "$outdir/${base}_hydro.xyz" \
      "$outdir/${base}_hydro.xyz_2" \
      "$outdir/${base}_energy.txt" \
      "$outdir/${base}_hydro.pdb_2" \
      "$outdir/${base}_hydro.xyz_2" \
      "$outdir/${base}_min.xyz" \
      "$outdir/${base}_min.seq"
# DONE
echo "✅ Done!"
echo "   Original PDB   : $input"
echo "   Minimized PDB  : $outdir/${base}_min.pdb"
echo "   Atom energies  : $outdir/${base}_energy.csv"
echo "   Residue avg    : $outdir/${base}_min.csv"


