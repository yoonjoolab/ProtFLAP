#!/bin/bash
# ========================================
# Protein Preprocessing with Tinker
# Usage: ./preprocess.sh -i input.pdb [-o output_folder]
# Output: minimized PDB, energy.csv, energy_avg.csv in output folder
# Intermediate files are deleted automatically
# ========================================


tinker_dir="/path/to/Tinker"
force="/path/to/force.key"
param_file="path/to/amber99sb.prm"       # Parameter file
min_grid=0.01                             # Minimization grid step

pdb2xyz="$tinker_dir/pdbxyz"
xyz2pdb="$tinker_dir/xyzpdb"
minimize="$tinker_dir/minimize"
analyze="$tinker_dir/analyze"

input=""
outdir=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input) shift; input="$1" ;;
        -o|--output) shift; outdir="$1" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$input" ]; then
    echo "Error: Input PDB file is required."
    exit 1
fi

input="$(realpath "$input")"
input_dir="$(dirname "$input")"
base="$(basename "$input" .pdb)"

if [ -z "$outdir" ]; then outdir="$input_dir"; else outdir="$(realpath "$outdir")"; fi
mkdir -p "$outdir"

echo "🔹 Processing PDB: $input"
echo "Output folder: $outdir"

pushd "$input_dir" > /dev/null

# STEP 1: Add hydrogens + prepare XYZ
"$pdb2xyz" "$base.pdb" -k "$force" "$param_file" ALL A ALL
"$xyz2pdb" "${base}.xyz" -k "$force" "$param_file"
mv "${base}.pdb_2" "$outdir/${base}_hydro.pdb"
"$pdb2xyz" "$outdir/${base}_hydro.pdb" -k "$force" "$param_file" ALL A ALL

# STEP 2: Minimize structure (version-agnostic)
# Try old syntax first; if fails, try new syntax
if ! "$minimize" "$outdir/${base}_hydro.xyz" -k "$force" "$param_file" "$min_grid"; then
    echo "⚠️ Old syntax failed, trying new Tinker syntax..."
    "$minimize" "$outdir/${base}_hydro.xyz" "$min_grid" -k "$force" "$param_file"
fi

"$xyz2pdb" "$outdir/${base}_hydro.xyz_2" -k "$force" "$param_file"
mv "$outdir/${base}_hydro.pdb_2" "$outdir/${base}_min.pdb"

# STEP 3: Energy breakdown
"$pdb2xyz" "$outdir/${base}_min.pdb" -k "$force" "$param_file" ALL A ALL
"$xyz2pdb" "$outdir/${base}_min.xyz" -k "$force" "$param_file"
"$analyze" "$outdir/${base}_min.xyz" -k "$force" "$param_file" A | \
awk '/Potential Energy Breakdown over Atoms :/{flag=1} flag {print}' > "$outdir/${base}_energy.txt"

popd > /dev/null


# STEP 4: Convert results to CSV
python3 preprocess_scripts/convert_to_csv.py "$outdir/${base}_min.pdb" "$outdir/${base}_energy.txt" "$outdir/${base}_energy.csv"
python3 preprocess_scripts/average_per_residue.py "$outdir/${base}_energy.csv" "$outdir/${base}_min.csv"

# === STEP 5: Cleanup intermediate files ===
rm -f "${base}.xyz" \
      "${base}.seq" \
      "$outdir/${base}_hydro.pdb" \
      "$outdir/${base}_hydro.seq" \
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


