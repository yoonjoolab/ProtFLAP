#!/bin/bash
# =====================================================
# Protein Preprocessing with Tinker (MULTI-PDB SAFE)
# ========================================
# Protein Preprocessing with Tinker
# Usage: ./preprocess.sh -i input.pdb [-o output_folder]
# Output: minimized PDB, energy.csv, energy_avg.csv in output folder
# Intermediate files are deleted automatically
# =====================================================

set -euo pipefail

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
    case "$1" in
        -i|--input)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                inputs+=("$1")
                shift
            done
            ;;
        -o|--output)
            shift
            outdir="$1"
            shift
            ;;
        *)
            echo "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "❌ Error: At least one input .pdb file is required"
    exit 1
fi

# =========================
# EXPAND WILDCARDS
# =========================
pdb_files=()
for item in "${inputs[@]}"; do
    pdb_files+=( $(ls $item 2>/dev/null || true) )
done

if [[ ${#pdb_files[@]} -eq 0 ]]; then
    echo "❌ No PDB files found"
    exit 1
fi

# =========================
# OUTPUT DIR
# =========================
if [[ -n "$outdir" ]]; then
    outdir="$(realpath "$outdir")"
    mkdir -p "$outdir"
fi

# =====================================================
# MAIN LOOP
# =====================================================
for pdb in "${pdb_files[@]}"; do

    if [[ ! "$pdb" =~ \.pdb$ ]]; then
        echo "⚠ Skipping non-PDB file: $pdb"
        continue
    fi

    pdb="$(realpath "$pdb")"
    input_dir="$(dirname "$pdb")"
    base="$(basename "$pdb" .pdb)"

    work_outdir="$outdir"
    if [[ -z "$work_outdir" ]]; then
        work_outdir="$input_dir"
    fi

    mkdir -p "$work_outdir"

    echo "====================================================="
    echo "🔹 Processing: $pdb"
    echo "   Output dir: $work_outdir"
    echo "====================================================="

    pushd "$input_dir" > /dev/null

    # -------------------------
    # STEP 1: Add Hydrogens
    # -------------------------
    "$pdb2xyz" "$base.pdb" -k "$force" "$param_file" ALL A ALL
    "$xyz2pdb" "${base}.xyz" -k "$force" "$param_file"

    mv "${base}.pdb_2" "$work_outdir/${base}_hydro.pdb"

    "$pdb2xyz" "$work_outdir/${base}_hydro.pdb" -k "$force" "$param_file" ALL A ALL

    # -------------------------
    # STEP 2: Minimization
    # -------------------------
    if ! "$minimize" "$work_outdir/${base}_hydro.xyz" -k "$force" "$param_file" "$min_grid"; then
        echo "⚠ Old minimize syntax failed → trying new syntax"
        "$minimize" "$work_outdir/${base}_hydro.xyz" "$min_grid" -k "$force" "$param_file"
    fi

    "$xyz2pdb" "$work_outdir/${base}_hydro.xyz_2" -k "$force" "$param_file"
    mv "$work_outdir/${base}_hydro.pdb_2" "$work_outdir/${base}_min.pdb"

    # -------------------------
    # STEP 3: Energy Breakdown
    # -------------------------
    "$pdb2xyz" "$work_outdir/${base}_min.pdb" -k "$force" "$param_file" ALL A ALL
    "$xyz2pdb" "$work_outdir/${base}_min.xyz" -k "$force" "$param_file"

    "$analyze" "$work_outdir/${base}_min.xyz" -k "$force" "$param_file" A | \
    awk '/Potential Energy Breakdown over Atoms :/{flag=1} flag {print}' \
    > "$work_outdir/${base}_energy.txt"

    popd > /dev/null

    # -------------------------
    # STEP 4: CSV CONVERSION
    # -------------------------
    python3 preprocess_scripts/convert_to_csv.py \
        "$work_outdir/${base}_min.pdb" \
        "$work_outdir/${base}_energy.txt" \
        "$work_outdir/${base}_energy.csv"

    python3 preprocess_scripts/average_per_residue.py \
        "$work_outdir/${base}_energy.csv" \
        "$work_outdir/${base}_min.csv"

    # -------------------------
    # STEP 5: CLEANUP
    # -------------------------
    rm -f \
        "${input_dir}/${base}.xyz" \
        "${input_dir}/${base}.seq" \
        "$work_outdir/${base}_hydro.pdb" \
        "$work_outdir/${base}_hydro.seq" \
        "$work_outdir/${base}_hydro.xyz" \
        "$work_outdir/${base}_hydro.xyz_2" \
        "$work_outdir/${base}_hydro.pdb_2" \
        "$work_outdir/${base}_min.xyz" \
        "$work_outdir/${base}_min.seq" \
        "$work_outdir/${base}_min.pdb_2" \
        "$work_outdir/${base}_energy.txt"

    echo "✅ Finished: $base"
    echo "   Minimized PDB : $work_outdir/${base}_min.pdb"
    echo "   Residue CSV  : $work_outdir/${base}_min.csv"
done

echo "🎉 All PDBs processed successfully."

