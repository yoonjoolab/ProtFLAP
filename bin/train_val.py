import os
import csv
import torch
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import subprocess
import re
import networkx as nx
import shutil
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, ParameterSampler
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, GATConv
from torch.optim.lr_scheduler import ReduceLROnPlateau


from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from Bio import PDB
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from aaindex import aaindex1

# ✅ Import from your modular files
from .gnn_model import NodeMLP_GCN, FocalLoss
from .evaluate import compute_metrics, evaluate_and_plot_confusion_matrix_per_residue

from .data_utils import load_rmsf_data
from .prediction_utils import get_predictions, compare_rmsf_and_predictions

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

def count_flexible_residues(dataset, threshold=0.0):
    flexible_count, non_flexible_count = 0, 0
    for data in dataset:
        rmsf_values = load_rmsf_data(data.pdb_name)
        if rmsf_values is None:
            continue
        for val in rmsf_values:
            if val > threshold:
                flexible_count += 1
            else:
                non_flexible_count += 1
    return flexible_count, non_flexible_count


# ADD THESE FUNCTIONS HERE, BEFORE get_predictions
def parse_freesasa_output(output):
    lines = output.strip().split("\n")
    parsed_data = {}

    for line in lines:
        if line.startswith("#") or not line.strip():
            continue  # Skip header lines or empty lines

        # Example line: SEQ A    1  GLN :  196.95
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 5:
            try:
                residue_number = int(re.sub(r"\D", "", parts[2]))  # Remove non-digit characters
                surface_area = float(parts[5].replace(":", "").strip())
                parsed_data[residue_number] = surface_area
            except ValueError:
                continue  # Skip lines with invalid residue numbers

    return parsed_data

# Constants
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
    'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

AMINO_ACIDS = 'GAVLMIWFYSTCPNQKRHDE'
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
POLARITY_CATEGORIES = {
    'nonpolar': ['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W'],
    'polar_uncharged': ['C', 'N', 'Q', 'S', 'T', 'Y'],
    'polar_positive': ['H', 'K', 'R'],
    'polar_negative': ['D', 'E']
}

def one_hot_encode(residue_name):
    index = AA_TO_INDEX.get(residue_name, -1)
    if index != -1:
        encoding = [0] * 20
        encoding[index] = 1
        return encoding
    else:
        return [0] * 20

def get_polarity_encoding(residue_name):
    for category, residues in POLARITY_CATEGORIES.items():
        if residue_name in residues:
            return [1 if category == cat else 0 for cat in POLARITY_CATEGORIES.keys()]
    return [0] * len(POLARITY_CATEGORIES)

def dssp_simplified_encode(dssp_value):
    # S: Helix (H, G, I)
    # E: Extended (E, B)
    # L: Loop (T, S, -, C)
    if dssp_value in ['H', 'G', 'I']:
        return [1, 0, 0]  # Helix
    elif dssp_value in ['E', 'B']:
        return [0, 1, 0]  # Extended
    else:
        return [0, 0, 1]  # Loop (including coil, bend, turn, and undefined)

def get_residue_features(residue, freesasa_value, atomic_energies, dssp_value):
    residue_name = THREE_TO_ONE.get(residue.get_resname(), 'X')

    # One-hot encode residue features
    one_hot = one_hot_encode(residue_name)
    polarity_encoding = get_polarity_encoding(residue_name)

    # Other residue features from aaindex1
    hydrophobicity_values = aaindex1['KYTJ820101']['values']
    size_values = aaindex1['FASG760101']['values']
    charge_values = aaindex1['KLEP840101']['values']

    hydrophobicity = hydrophobicity_values.get(residue_name, 0)
    size = size_values.get(residue_name, 0)
    charge = charge_values.get(residue_name, 0)

    # Apply FreeSASA threshold: 1 if >= 15 else 0
    exposure = 1 if freesasa_value >= 15 else 0

    # DSSP structural encoding
    dssp_encoding = dssp_simplified_encode(dssp_value)

    # Combine all features into a single vector
    feature_vector = one_hot + polarity_encoding + [
        hydrophobicity, size, charge, exposure
    ] + dssp_encoding + atomic_energies

    return feature_vector

def move_mismatched_files(pdb_file, csv_file, destination_directory="mismatched_files"):
    """Moves the specified PDB and CSV files to the destination directory."""
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    pdb_name = os.path.basename(pdb_file)
    csv_name = os.path.basename(csv_file)
    
    pdb_destination = os.path.join(destination_directory, pdb_name)
    csv_destination = os.path.join(destination_directory, csv_name)

    try:
        shutil.move(pdb_file, pdb_destination)
        logging.info(f"Moved {pdb_name} to {destination_directory}")
    except Exception as e:
        logging.error(f"Error moving {pdb_name} to {destination_directory}: {e}")
        
    try:
        shutil.move(csv_file, csv_destination)
        logging.info(f"Moved {csv_name} to {destination_directory}")
    except Exception as e:
        logging.error(f"Error moving {csv_name} to {destination_directory}: {e}")

def load_and_present_pdb(pdb_file):
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
    atomic_energy_csv = f"{pdb_name}.csv"

    if not os.path.exists(atomic_energy_csv):
        logging.error(f"Required CSV file not found for {pdb_name}: {atomic_energy_csv}")
        return None

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("structure", pdb_file)
    except Exception as e:
        logging.error(f"PDB parsing error for {pdb_name}: {e}")
        move_mismatched_files(pdb_file, atomic_energy_csv)
        return None

    try:
        atomic_energy_data = pd.read_csv(atomic_energy_csv)
    except Exception as e:
        logging.error(f"Error reading atomic energy CSV for {pdb_name}: {e}")
        move_mismatched_files(pdb_file, atomic_energy_csv)
        return None

    # Run freesasa and parse output
    freesasa_command = f"freesasa --foreach-residue --no-log {pdb_file}"
    try:
        freesasa_result = subprocess.run(freesasa_command, shell=True, capture_output=True, text=True, check=True)
        freesasa_output = freesasa_result.stdout
        freesasa_dict = parse_freesasa_output(freesasa_output)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running freesasa for {pdb_name}: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"Error parsing freesasa output for {pdb_name}: {e}")
        return None

    # Clean and convert 'Residue' column in atomic_energy_data
    def clean_and_convert_residue(residue):
        residue = str(residue).strip().replace(',', '')
        try:
            return int(residue)
        except ValueError:
            return None

    atomic_energy_data["Residue"] = atomic_energy_data["Residue"].apply(clean_and_convert_residue)
    atomic_energy_data = atomic_energy_data.dropna(subset=["Residue"])
    atomic_energy_dict = dict(zip(atomic_energy_data["Residue"], atomic_energy_data.iloc[:, 1:29].values.tolist()))

    dssp_dict = dssp_dict_from_pdb_file(pdb_file)[0]

    all_node_features = []
    protein_graphs = {}
    node_features_dict = {}
    residue_counter = 0
    pdb_residue_count = 0

    for model in structure:
        for chain in model:
            residues_list = [residue for residue in chain if PDB.is_aa(residue)]
            chain_node_features = []
            chain_node_features_dict = {}
            protein_graph = create_protein_graph(structure, chain.get_id())
            protein_graphs[chain.get_id()] = protein_graph

            for idx, residue in enumerate(residues_list):
                pdb_residue_count += 1
                try:
                    residue_id = residue.get_id()
                    residue_number = residue_id[1]
                    freesasa_value = freesasa_dict.get(residue_number, 0)
                    atomic_energies = atomic_energy_dict.get(residue_number, [0] * 28)

                    dssp_value = dssp_dict.get((chain.get_id(), residue_id), '-')
                    features = get_residue_features(residue, freesasa_value, atomic_energies, dssp_value)
                    chain_node_features.append(features)
                    chain_node_features_dict[residue_counter] = features
                    residue_counter += 1
                except Exception as e:
                    logging.warning(f"Error processing residue {residue_id}: {e}")
                    continue
            all_node_features.extend(chain_node_features)
            node_features_dict[chain.get_id()] = chain_node_features_dict
            
    csv_residue_count = len(atomic_energy_data)

    if not all_node_features:
        logging.error(f"No valid residues found in {pdb_name}")
        move_mismatched_files(pdb_file, atomic_energy_csv)
        return None
        
    if pdb_residue_count != csv_residue_count:
        logging.warning(f"Mismatch in number of residues for {pdb_name}: PDB has {pdb_residue_count}, CSV has {csv_residue_count}")
        move_mismatched_files(pdb_file, atomic_energy_csv)
        return None

    node_features_tensor = torch.tensor(all_node_features, dtype=torch.float)
    logging.info(f"Processed {pdb_name}")
    print(f"Processed PDB: {pdb_name}, Overall node dimension: {node_features_tensor.shape[1]}")
    return node_features_tensor, protein_graphs, node_features_dict


def create_protein_graph(structure, chain_id, max_distance=30.0):
    graph = nx.Graph()
    chain = structure[0][chain_id]
    residues = [residue for residue in chain if PDB.is_aa(residue)]
    for idx, residue in enumerate(residues):
        graph.add_node(idx, residue=residue)
    for i, res1 in enumerate(residues):
        for j, res2 in enumerate(residues[i+1:], start=i+1):
            distance = np.linalg.norm(res1['CA'].coord - res2['CA'].coord)
            if distance < max_distance:
                graph.add_edge(i, j, distance=distance)
    return graph

def build_dataset_from_folder(folder: str):
    data_list = []
    for filename in os.listdir(folder):
        if not filename.endswith(".pdb"):
            continue
        pdb_path = os.path.join(folder, filename)
        pdb_name = os.path.splitext(filename)[0]
        result = load_and_present_pdb(pdb_path)
        if result is None:
            continue
        node_features_tensor, protein_graphs, node_features_dict = result
        csv_file = os.path.join(folder, f"{pdb_name}.csv")
        if not os.path.exists(csv_file):
            continue
        df = pd.read_csv(csv_file)
        if "rmsf_norm" not in df.columns:
            continue
        rmsf = df["rmsf_norm"].values.astype(np.float32)
        labels = (rmsf > 0).astype(np.float32)  # threshold 0 as requested
        scaler = StandardScaler()
        node_features_tensor = torch.tensor(scaler.fit_transform(node_features_tensor), dtype=torch.float)
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
            L = min(x.shape[0], len(labels))
            if L == 0:
                continue
            x = x[:L]
            y = torch.tensor(labels[:L], dtype=torch.float)
            edge_index = torch.clamp(edge_index, min=0, max=L - 1)
            data = Data(x=x, edge_index=edge_index, y=y)
            data.pdb_name = pdb_name
            data_list.append(data)
    return data_list

def train_and_evaluate(trial, data_list):
    # Suggest hyperparameters with Optuna
    lr = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    dropout = trial.suggest_float("dropout", 0.001, 0.3)
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2, 1e-1])
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    residual = trial.suggest_categorical("residual", [True, False])
    activation = trial.suggest_categorical("activation", ["ReLU", "Tanh", "LeakyReLU", "ELU"])
    use_bias = trial.suggest_categorical("use_bias", [True, False])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_gcn_layers = trial.suggest_int("num_gcn_layers", 2, 6)

    n = len(data_list)
    n_train = int(0.8 * n)
    train_ds, val_ds = random_split(data_list, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=Batch.from_data_list)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=True, collate_fn=Batch.from_data_list)

    in_node_feats = data_list[0].x.size(1)

    model = NodeMLP_GCN(
        in_node_feats,
        hidden_dim=hidden_dim,
        num_gcn_layers=num_gcn_layers,
        dropout=dropout,
        use_residual=residual,
        use_batch_norm=batch_norm,
        activation=activation,
        use_bias=use_bias,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)

    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    for epoch in range(1, 101):
        model.train()
        total_train_loss, total_train_nodes = 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_nodes
            total_train_nodes += batch.num_nodes
        train_loss = total_train_loss / total_train_nodes

        model.eval()
        total_val_loss, total_val_nodes = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch.x, batch.edge_index)
                loss = criterion(logits, batch.y.float())
                total_val_loss += loss.item() * batch.num_nodes
                total_val_nodes += batch.num_nodes
        val_loss = total_val_loss / total_val_nodes

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_val_loss

def train_one_fold(
    fold_id, train_idx, val_idx, dataset, best_params, epochs=100, batch_size=16, patience=20
):
    train_ds = [dataset[i] for i in train_idx]
    val_ds = [dataset[i] for i in val_idx]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list
    )

    in_node_feats = dataset[0].x.size(1)

    model = NodeMLP_GCN(
        in_node_feats,
        hidden_dim=best_params["hidden_dim"],
        num_gcn_layers=best_params["num_gcn_layers"],
        dropout=best_params["dropout"],
        use_residual=best_params["residual"],
        use_batch_norm=best_params["batch_norm"],
        activation=best_params["activation"],
        use_bias=best_params["use_bias"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )
    criterion = FocalLoss(alpha=0.5, gamma=2.0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=5, verbose=True
    )

    best_train_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    same_loss_counter = 0  # <-- to track repeated loss
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # === Training ===
        model.train()
        total_train_loss, total_train_nodes = 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_nodes
            total_train_nodes += batch.num_nodes
        train_loss = total_train_loss / total_train_nodes

        # === Validation (monitoring only) ===
        model.eval()
        total_val_loss, total_val_nodes = 0, 0
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch.x, batch.edge_index)
                loss = criterion(logits, batch.y.float())
                total_val_loss += loss.item() * batch.num_nodes
                total_val_nodes += batch.num_nodes
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_preds.append((probs > 0.5).cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
        val_loss = total_val_loss / total_val_nodes

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        acc, f1, mcc, precision, recall, fpr, tpr, roc_auc = compute_metrics(
            all_targets, all_preds, all_probs
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(train_loss)  # reduce LR based on training loss

        print(
            f"[Fold {fold_id}] Epoch {epoch:03d} | Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}"
        )

        # === Save best model based on training loss ===
        if train_loss < best_train_loss:
            if train_loss == best_train_loss:  
                same_loss_counter += 1
            else:
                same_loss_counter = 0  # reset if new improvement

            best_train_loss = train_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # === Early stopping conditions ===
        if patience_counter >= patience:
            print(f"[Fold {fold_id}] Early stopping (patience {patience}) at epoch {epoch}")
            break
        if same_loss_counter >= 20:  # <-- stop if no improvement for 20 epochs
            print(f"[Fold {fold_id}] Early stopping (no improvement for 20 epochs) at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f"best_model_fold{fold_id}.pth")

    print(f"[Fold {fold_id}] Best train loss: {best_train_loss:.4f}")

    # === Plot Loss Curve ===
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title(f"Loss Curve Fold {fold_id}")
    plt.savefig(f"loss_curve_fold{fold_id}.png")
    plt.close()

    # === Save report ===
    with open(f"classification_report_fold{fold_id}.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"AUC: {roc_auc:.4f}\n")

    evaluate_and_plot_confusion_matrix_per_residue(
        model,
        val_loader,
        device=device,
        load_rmsf_fn=load_rmsf_data,
        fold_id=fold_id,
    )

    return best_train_loss, f"best_model_fold{fold_id}.pth", (fpr, tpr, roc_auc)


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

if __name__ == "__main__":
    folder = os.getcwd()
    dataset = build_dataset_from_folder(folder)
    print(f"Built {len(dataset)} protein graphs.")

    # Count and print total flexible and non-flexible residues before training
    flexible_count, non_flexible_count = count_flexible_residues(dataset)
    print(f"Total flexible residues: {flexible_count}")
    print(f"Total non-flexible residues: {non_flexible_count}")

    # Create Optuna study and optimize hyperparameters
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: train_and_evaluate(trial, dataset), n_trials=20)  # or more trials

    trial = study.best_trial
    print("Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Save best hyperparameters and info to a text file
    with open("best_hyperparameters.txt", "w") as f:
        f.write("Best trial hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest trial value (lowest validation loss): {trial.value:.4f}\n")

    # Run cross-validation with the best found parameters
    run_cross_validation(dataset, trial.params, n_splits=5)

