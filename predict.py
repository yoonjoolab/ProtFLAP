import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Batch
from bin.train_val import load_and_present_pdb
from bin.gnn_model import NodeMLP_GCN

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FILE = "models/best_model_cv.pth"

default_params = {
    "learning_rate": 0.001,
    "dropout": 0.03773417328415976,
    "weight_decay": 0.0001,
    "batch_norm": False,
    "residual": True,
    "activation": "ReLU",
    "use_bias": True,
    "hidden_dim": 64,
    "num_gcn_layers": 5
}

def build_dataset_from_single_pdb_no_labels(pdb_file: str):
    """
    Build dataset from a single PDB file for prediction.

    Automatically looks for the CSV in the same folder as pdb_file by temporarily
    changing the working directory.
    """
    import contextlib

    pdb_file = os.path.abspath(pdb_file)
    pdb_folder = os.path.dirname(pdb_file)
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]

    # Check CSV exists in the same folder
    csv_file = os.path.join(pdb_folder, f"{pdb_name}.csv")
    if not os.path.exists(csv_file):
        print(f"ERROR: Required CSV file not found for {pdb_name}: {csv_file}")
        return []

    # Temporarily switch to PDB folder so load_and_present_pdb finds CSV
    with contextlib.ExitStack() as stack:
        old_cwd = os.getcwd()
        os.chdir(pdb_folder)
        try:
            result = load_and_present_pdb(pdb_file)
        finally:
            os.chdir(old_cwd)

    if result is None:
        print(f"No data returned by load_and_present_pdb for {pdb_name}")
        return []

    node_features_tensor, protein_graphs, node_features_dict = result

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    node_features_tensor = torch.tensor(scaler.fit_transform(node_features_tensor), dtype=torch.float)

    data_list = []

    for chain_id, G in protein_graphs.items():
        n = G.number_of_nodes()
        if n == 0:
            continue

        x = torch.tensor([node_features_dict[chain_id][i] for i in range(n)], dtype=torch.float)
        x = torch.tensor(scaler.fit_transform(x), dtype=torch.float)

        src_list, dst_list = [], []
        coords = np.array([node_features_dict[chain_id][i][:3] for i in range(n)])
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= 30.0:
                    src_list.extend([i, j])
                    dst_list.extend([j, i])

        if len(src_list) == 0:
            edge_index = torch.arange(n, dtype=torch.long).repeat(2, 1)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.pdb_name = pdb_name
        data_list.append(data)

    return data_list


def predict_single_pdb_no_labels(pdb_file, outdir=None):
    """Predict RMSF-like values and save CSV in the output folder (default: same as PDB)."""
    dataset = build_dataset_from_single_pdb_no_labels(pdb_file)
    if len(dataset) == 0:
        print(f"No data found for PDB: {pdb_file}")
        return

    in_feats = dataset[0].x.size(1)

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
        logits, probs = model(batch.x, batch.edge_index)
        probs_np = probs.cpu().numpy()
        binary_preds = (probs >= 0.5).cpu().numpy().astype(np.float32)
        residues = np.arange(batch.num_nodes)

    # Determine output folder
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(pdb_file))
    else:
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)

    output_file = os.path.join(outdir, os.path.splitext(os.path.basename(pdb_file))[0] + "_predictions.csv")
    df = pd.DataFrame({
        "residue_index": residues,
        "predicted_prob": probs_np,
        "predicted_binary": binary_preds
    })
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Predict RMSF-like values for PDB files using NodeMLP_GCN.")
    parser.add_argument("-i", "--input", required=True, help="Input PDB file (CSV must be in the same folder).")
    parser.add_argument("-o", "--output", help="Output folder for predictions (default: same as PDB).")
    args = parser.parse_args()

    # Expand wildcards if needed
    pdb_files = glob.glob(args.input)

    for pdb_file in pdb_files:
        predict_single_pdb_no_labels(pdb_file, args.output)

