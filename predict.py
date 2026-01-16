import os
import glob
import torch
import argparse
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler

from bin.train_val import load_and_present_pdb
from bin.gnn_model import NodeMLP_GCN

# =====================================================
# CONFIG
# =====================================================
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FILE = "models/best_model_cv.pth"

default_params = {
    "learning_rate": 0.0001,
    "dropout": 0.06837534221738543,
    "weight_decay": 0.0001,
    "batch_norm": False,
    "residual": True,
    "activation": "ReLU",
    "use_bias": True,
    "hidden_dim": 128,
    "num_gcn_layers": 4
}
# =====================================================
# DATASET BUILDER (NO LABELS)
# =====================================================
def build_dataset_from_single_pdb_no_labels(pdb_file: str):
    """
    Build graph dataset from a single PDB file.
    Requires a CSV with the SAME basename in the SAME directory.
    """

    pdb_file = os.path.abspath(pdb_file)

    if not pdb_file.lower().endswith(".pdb"):
        raise ValueError(f"Input must be a .pdb file: {pdb_file}")

    pdb_dir = os.path.dirname(pdb_file)
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
    csv_file = os.path.join(pdb_dir, f"{pdb_name}.csv")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Required CSV not found: {csv_file}")

    # Ensure CSV is discoverable by load_and_present_pdb
    old_cwd = os.getcwd()
    os.chdir(pdb_dir)

    try:
        result = load_and_present_pdb(pdb_file)
    finally:
        os.chdir(old_cwd)

    if result is None:
        return []

    node_features_tensor, protein_graphs, node_features_dict = result

    scaler = StandardScaler()
    node_features_tensor = torch.tensor(
        scaler.fit_transform(node_features_tensor),
        dtype=torch.float
    )

    data_list = []

    for chain_id, G in protein_graphs.items():
        n = G.number_of_nodes()
        if n == 0:
            continue

        x = torch.tensor(
            [node_features_dict[chain_id][i] for i in range(n)],
            dtype=torch.float
        )
        x = torch.tensor(scaler.fit_transform(x), dtype=torch.float)

        coords = np.array(
            [node_features_dict[chain_id][i][:3] for i in range(n)]
        )

        src, dst = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(coords[i] - coords[j]) <= 30.0:
                    src += [i, j]
                    dst += [j, i]

        if len(src) == 0:
            edge_index = torch.arange(n).repeat(2, 1)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.pdb_name = pdb_name
        data.chain_id = chain_id
        data_list.append(data)

    return data_list


# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_single_pdb_no_labels(pdb_file, outdir=None):

    dataset = build_dataset_from_single_pdb_no_labels(pdb_file)

    if len(dataset) == 0:
        print(f"⚠ No valid data found for {pdb_file}")
        return

    in_feats = dataset[0].x.shape[1]

    model = NodeMLP_GCN(
        in_node_feats=in_feats,
        hidden_dim=default_params["hidden_dim"],
        num_gcn_layers=default_params["num_gcn_layers"],
        dropout=default_params["dropout"],
        use_residual=default_params["residual"],
        use_batch_norm=default_params["batch_norm"],
        activation=default_params["activation"],
        use_bias=default_params["use_bias"],
    ).to(device)

    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    batch = Batch.from_data_list(dataset).to(device)

    with torch.no_grad():
        _, probs = model(batch.x, batch.edge_index)

    probs = probs.cpu().numpy()
    binary = (probs >= 0.5).astype(np.int32)
    residues = np.arange(len(probs))

    # Output directory
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(pdb_file))
    else:
        os.makedirs(outdir, exist_ok=True)

    out_csv = os.path.join(
        outdir,
        os.path.splitext(os.path.basename(pdb_file))[0] + "_predictions.csv"
    )

    pd.DataFrame({
        "residue_index": residues,
        "predicted_prob": probs,
        "predicted_binary": binary
    }).to_csv(out_csv, index=False)

    print(f"✅ Saved predictions → {out_csv}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict RMSF-like flexibility from PDB files using NodeMLP_GCN"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        nargs="+",
        help="One or more .pdb files (wildcards allowed, e.g. *.pdb)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: same directory as each PDB)"
    )

    args = parser.parse_args()

    # Expand wildcards
    pdb_files = []
    for item in args.input:
        pdb_files.extend(glob.glob(item))

    pdb_files = sorted(set(pdb_files))

    if not pdb_files:
        raise SystemExit("❌ No PDB files found.")

    # Enforce .pdb only
    invalid = [f for f in pdb_files if not f.lower().endswith(".pdb")]
    if invalid:
        raise SystemExit(f"❌ Invalid inputs (only .pdb allowed): {invalid}")

    for pdb in pdb_files:
        predict_single_pdb_no_labels(pdb, args.output)

