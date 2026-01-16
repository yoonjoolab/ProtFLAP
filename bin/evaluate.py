import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    matthews_corrcoef, precision_score, recall_score,
    roc_curve, auc
)


def compute_metrics(all_targets, all_preds, all_probs):
    """
    Compute binary classification metrics + ROC curve.
    """
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    mcc = matthews_corrcoef(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    return acc, f1, mcc, precision, recall, fpr, tpr, roc_auc


def evaluate_and_plot_confusion_matrix_per_residue(model, data_loader, device, load_rmsf_fn, fold_id=None):
    """
    Evaluate model per residue and save CSV + confusion matrix.
    
    Args:
        model: trained PyTorch model
        data_loader: PyTorch Geometric DataLoader
        device: "cpu" or "cuda"
        load_rmsf_fn: function to load rmsf_norm values (you already have load_rmsf_data)
        fold_id: int or None
    """
    model.eval()
    all_preds, all_labels, all_probs, all_logits = [], [], [], []
    all_pdb_names, all_rmsf_values = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            for graph in batch.to_data_list():
                logits, probs = model(graph.x.to(device), graph.edge_index.to(device))
                preds = (probs > 0.5).float().cpu().numpy().flatten()
                labels = graph.y.cpu().numpy().astype(int).flatten()
                pdb_name = getattr(graph, "pdb_name", "Unknown")

                # Get rmsf values
                rmsf_values = load_rmsf_fn(pdb_name)
                n = len(labels)
                if rmsf_values is not None and len(rmsf_values) >= n:
                    use_rmsf = rmsf_values[:n]
                elif rmsf_values is not None:
                    use_rmsf = np.pad(rmsf_values, (0, n - len(rmsf_values)), constant_values=np.nan)
                else:
                    use_rmsf = [None] * n

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_logits.extend(logits.cpu().numpy().flatten())
                all_rmsf_values.extend(use_rmsf)
                all_pdb_names.extend([pdb_name] * n)

    # Save CSV
    csv_name = f"classification_results_per_residue_fold{fold_id}.csv" if fold_id else "classification_results_per_residue.csv"
    with open(csv_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "PDB_Name", "Residue_Index", "Actual_Binary", "Predicted_Binary",
            "Sigmoid_Output", "Logit", "rmsf_norm"
        ])
        for i, (pdb_name, label, pred, prob, logit, rmsf) in enumerate(
            zip(all_pdb_names, all_labels, all_preds, all_probs, all_logits, all_rmsf_values)
        ):
            writer.writerow([pdb_name, i, label, pred, prob, logit, rmsf])

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix Per Residue {'Fold ' + str(fold_id) if fold_id else ''}")
    plt.savefig(f"confusion_matrix_per_residue_fold{fold_id}.png" if fold_id else "confusion_matrix_per_residue.png")
    plt.close()

    return cm, all_labels, all_preds, all_probs, all_logits, all_rmsf_values
    
def run_cross_validation(dataset, best_params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    plt.figure(figsize=(8, 6))
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
        print(f"Fold {fold_id}: {len(train_idx)} training graphs, {len(val_idx)} validation graphs")
        best_val_loss, model_path, roc_data = train_one_fold(fold_id, train_idx, val_idx, dataset, best_params)
        fpr, tpr, roc_auc = roc_data
        plt.plot(fpr, tpr, label=f"Fold {fold_id} (AUC={roc_auc:.4f})")
        fold_results.append((best_val_loss, model_path))
    best_fold = min(fold_results, key=lambda x: x[0])
    os.rename(best_fold[1], "best_model_cv.pth")
    print(f"Best model selected: {best_fold[1]} -> saved as best_model_cv.pth")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for 5-Fold Cross-Validation")
    plt.legend()
    plt.savefig("roc_curves_5fold.png")
    plt.close()
