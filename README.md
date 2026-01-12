# ProtFLAP
Prediction of protein flexibility
---

## System Requirements

- **Operating System:** Ubuntu 20.04 recommended (other compatible Linux distributions may work) 
- **Python:** 3.10+ 
- **CUDA Toolkit:** 12.2 (if using GPU, other compatible versions are fine).

> Simply editing the scripts and installation steps will allow GPU support if desired.

---

## Installation

### Step 1. Install Python requirements
All Python dependencies are listed in `requirements.txt`. 
Install them using the helper script:

```bash
./basic_install.sh requirements.txt

```
### Step 2. Install FreeSASA

ProtFlap uses FreeSASA for solvent-accessible surface area calculations.

1. Download FreeSASA from GitHub:
👉 https://github.com/mittinatten/freesasa

2. Build and install FreeSASA:
3. Add FreeSASA to your system PATH (if needed):

```
export PATH=$PATH:/path/to/freesasa/bin
```
4. Verify the installation:
```
freesasa --version
```

### Step 3. Install Tinker
ProtFlap requires the **Tinker molecular modeling package** for energy minimization and structure preprocessing. 

Download Tinker from the official website: 
👉 [https://dasher.wustl.edu/tinker/](https://dasher.wustl.edu/tinker/)

**Setup instructions:**

1. Compile Tinker following the instructions on the website. 
2. Place your `amber99sb.prm` parameter file inside your Tinker folder. 
3. Create your `force.key` file for the force field. **Include the following options in `force.key`:**

```
parameters /path/to/amber99sb.prm
forcefield /path/to/amber99sb.prm

openmp-threads 1

verbose

SOLVATE GB-HPMF
```
4. Update the paths in `preprocess.py` to point to your Tinker executables, `force.key`, and parameter file:

```
tinker="path/to/tinker/"
force="path/to/force.key"
param_file="path/to/amber99sb.prm"

```

### Preprocessing PDB Files

All raw PDB files should be placed.
To preprocess a PDB file:

```
python preprocess.py -i protein.pdb -o output_folder

```

Arguments:

-i → Input file: Path to the PDB file you want to preprocess (e.g., protein.pdb).

-o → Output folder: Name of the folder where all results will be saved (e.g., output_folder). The folder will be created automatically if it doesn’t exist.

What the script does:

Adds hydrogens and converts the PDB to Tinker XYZ format.

Minimizes the structure using the Amber99SB force field.

Computes per-atom and per-residue energy breakdowns.

Outputs:

input_energy.csv → per-atom energies

input_min.csv → per-residue averaged energies

✅ The original PDB file is preserved, and all intermediate files are automatically removed.


### Running Predictions

After preprocessing, run ProtFlap prediction on the minimized PDB:

```
python predict.py -i protein_min.pdb -o predictions

```
### Output

Arguments:

-i  → Required. Path to the minimized PDB file.

-o  → Optional. Folder where the prediction CSV will be saved.

If not provided, the output CSV will be automatically saved in the same folder as the input PDB.

The output folder will be created automatically if it does not exist.
 
The prediction CSV contains three columns:

residue → Index of the residue in the protein chain

probability → Predicted probability of flexibility (sigmoid output)

predicted_binary → Binary classification (0 or 1) based on the threshold

The predicted_prob column can be used for regression-style plots or further analysis, while predicted_binary is for classification-based analysis.

The predicted_binary column is always provided for classification-based analysis.

## Quickstart Example

This example demonstrates a full run from raw PDB → preprocessing → prediction using the provided `5m99A02.pdb`.

### Step 1. Preprocess the PDB

Place the raw PDB file in the folder (already included: `15m99A02.pdb`).

Run the preprocessing script:

```
python preprocess.py -i 5m99A02.pdb -o example

```
This will:

Add hydrogens and convert the PDB to Tinker XYZ format

Minimize the structure using the Amber99SB force field
Compute per-atom and per-residue energy breakdowns

Generate two output CSVs in the example folder:

5m99A02_energy.csv → per-atom energies

5m99A02_min.csv → per-residue averaged energies

The minimized PDB will be saved as 5m99A02_min.pdb

### Step 2. Run ProtFlap Prediction

Use the minimized PDB as input for the prediction:

```
python predict.py example/5m99A02_min.pdb -o example/predictions

```
The output CSV (e.g., 5m99A02_min_predictions.csv) 

**Notes**

Always use the _min.pdb produced by preprocess.sh as input.

The script outputs residue-level flexibility predictions.

GPU support can be enabled if compatible hardware and CUDA versions are installed.







