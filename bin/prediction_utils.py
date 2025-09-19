import numpy as np
import pandas as pd
import torch
from .data_utils import load_rmsf_data


def get_predictions(model, loader, device="cpu"):
    """
    Run model inference and return true labels, predictions, and pdb names.
    """
    model.eval()
    y_true, y_pred_sigmoid, y_pred_logits, pdb_names = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, sigmoid_output = model(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                getattr(batch, "batch", None)
            )
            y_true.extend(batch.y.cpu().numpy().flatten())
            y_pred_sigmoid.extend(sigmoid_output.cpu().numpy().flatten())
            y_pred_logits.extend(logits.cpu().numpy().flatten())
            batch_pdb_name = getattr(batch, "pdb_name", "Unknown")
            if isinstance(batch_pdb_name, (list, np.ndarray)):
                pdb_names.extend(batch_pdb_name)
            else:
                pdb_names.extend([batch_pdb_name] * batch.num_nodes)
    return np.array(y_true), np.array(y_pred_sigmoid), np.array(y_pred_logits), pdb_names


def compare_rmsf_and_predictions(pdb_names, y_true, y_pred_sigmoid, y_pred_logits):
    """
    Compare predicted values with actual rmsf_norm for each PDB.
    """
    results = []
    unique_pdbs = sorted(set(pdb_names))
    for pdb_name in unique_pdbs:
        rmsf_norm_values = load_rmsf_data(pdb_name)
        if rmsf_norm_values is None:
            continue
        idxs = [i for i, name in enumerate(pdb_names) if name == pdb_name]
        for i, idx in enumerate(idxs):
            results.append({
                "PDB Name": pdb_name,
                "Residue_Index": i,
                "rmsf_norm": rmsf_norm_values[i] if i < len(rmsf_norm_values) else None,
                "True Label": y_true[idx],
                "Sigmoid Output": y_pred_sigmoid[idx],
                "Logit": y_pred_logits[idx]
            })
    return pd.DataFrame(results)

