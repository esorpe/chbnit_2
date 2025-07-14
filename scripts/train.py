# # scripts/train.py
# """
# Train EEGNet on preprocessed CHBNIT data (raw bandpass or gaussian).
# Usage:
#     python scripts/train.py --filter bandpass
#     python scripts/train.py --filter gaussian
# """
# import os
# import glob
# import argparse
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
# from models.eegnet import EEGNet



# # --- EEGDataset loader for .npz files ---
# class EEGDataset(Dataset):
#     def __init__(self, file_list):
#         self.files = file_list

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         data = np.load(self.files[idx], mmap_mode='r')
#         # Assuming each .npz file contains 'X' and 'Y' arrays
    
#         X = data['X']  # (n_epochs, n_channels, n_times)
#         Y = data['Y']  # (n_epochs,)
#         # We flatten file-level arrays into samples
#         # but better: each file contains many epochs; handle per-epoch elsewhere
#         # Here we return entire arrays
#         return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)


# def collate_fn(batch):
#     # batch: list of (X_arr, Y_arr)
#     Xs, Ys = zip(*batch)
#     X = torch.cat([x for x in Xs], dim=0)  # (sum n_epochs, n_channels, n_times)
#     Y = torch.cat([y for y in Ys], dim=0)
#     # Add channel dimension for EEGNet: (N, 1, C, T)
#     X = X.unsqueeze(1)
#     return X, Y


# def compute_metrics(y_true, y_pred, y_proba=None):
#     acc = accuracy_score(y_true, y_pred)
#     auroc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
#     cm = confusion_matrix(y_true, y_pred, labels=[0,1])
#     return acc, auroc, cm


# def train_epoch(model, loader, optimizer, loss_fn, device):
#     model.train()
#     total_loss = 0
#     all_preds, all_labels, all_proba = [], [], []
#     for X, Y in loader:
#         X, Y = X.to(device), Y.to(device)
#         optimizer.zero_grad()
#         out = model(X)
#         proba = nn.functional.softmax(out, dim=1)[:,1]
#         loss = loss_fn(out, Y)
#         loss.backward()
#         optimizer.step()
#         preds = out.argmax(dim=1)
#         total_loss += loss.item() * Y.size(0)
#         all_preds.append(preds.cpu())
#         all_labels.append(Y.cpu())
#         all_proba.append(proba.cpu())
#     y_pred = torch.cat(all_preds).numpy()
#     y_true = torch.cat(all_labels).numpy()
#     y_proba = torch.cat(all_proba).numpy()
#     avg_loss = total_loss / len(y_true)
#     acc, auroc, cm = compute_metrics(y_true, y_pred, y_proba)
#     return avg_loss, acc, auroc, cm


# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--filter', choices=['bandpass','gaussian'], default='bandpass')
# #     parser.add_argument(
# #         '--max-files',
# #         type=int,
# #         default=None,
# #         help='If set, only use this many .npz files for training+val splitting'

# #     )
    
# #     parser.add_argument('--batch', type=int, default=64)
# #     parser.add_argument('--epochs', type=int, default=20)
# #     args = parser.parse_args()

# #     # data_dir = f"/Volumes/Samsung_T5/project1/data/epochs_{args.filter}"
# #     project_root = os.path.dirname(os.path.dirname(__file__))
# #     data_dir = os.path.join(project_root, 'data', f'epochs_{args.filter}')
# #     RAW_ROOT = "edf"
# #     if not os.path.exists(data_dir):
# #         raise FileNotFoundError(f"Data directory {data_dir} does not exist. Run preprocessing first.")
# #     all_files = glob.glob(os.path.join(data_dir, '*', f"{args.filter}_epochs_*.npz"))
# #     print("looking for files in", data_dir)
# #     print("Matched files:", all_files)

# #     if args.max_files is not None:
# #         all_files = all_files[: args.max_files]


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--filter', choices=['bandpass','gaussian'], default='bandpass')
#     parser.add_argument('--max-files', type=int, default=None)
#     parser.add_argument('--batch', type=int, default=64)
#     parser.add_argument('--epochs', type=int, default=20)
#     args = parser.parse_args()

#     # Compute project root and data directory
#     project_root = os.path.dirname(os.path.dirname(__file__))
#     data_dir     = os.path.join(project_root, "data", f"epochs_{args.filter}")

#     print("Training will load from:", data_dir)
#     all_files = glob.glob(os.path.join(data_dir, '*', f"{args.filter}_epochs_*.npz"))
#     if args.max_files:
#         all_files = all_files[: args.max_files]

#     print("Matched files:", all_files)
#     if not all_files:
#         raise FileNotFoundError(f"No .npz files found in {data_dir}. Did you run preprocessing?")    
    
#     train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

#     train_ds = EEGDataset(train_files)
#     val_ds = EEGDataset(val_files)
#     train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
#     val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = EEGNet().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     # weight classes inversely to handle imbalance
#     loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=device))

#     for epoch in range(1, args.epochs+1):
#         train_loss, train_acc, train_auroc, _ = train_epoch(model, train_loader, optimizer, loss_fn, device)
#         val_loss, val_acc, val_auroc, _     = train_epoch(model, val_loader,   optimizer, loss_fn, device)
#         print(f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.4f}, AUROC={train_auroc:.4f} | \
#               Val loss={val_loss:.4f}, acc={val_acc:.4f}, AUROC={val_auroc:.4f}")

#     # Save model
#     os.makedirs('models/saved', exist_ok=True)
#     torch.save(model.state_dict(), f"models/saved/eegnet_{args.filter}.pth")

# if __name__ == '__main__':
#     main()


# scripts/train.py
"""
Train EEGNet on preprocessed CHBNIT data (raw bandpass or gaussian).
Usage:
    python scripts/train.py --filter bandpass
    python scripts/train.py --filter gaussian
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from models.eegnet import EEGNet
import gc
from tqdm import tqdm



# # --- EEGDataset optimized for minimal memory (per-epoch access) ---
# class EEGDataset(Dataset):
#     def __init__(self, file_list):
#         self.epoch_index = []  # List of (file_path, epoch_idx)
#         self.file_data = {}    # Cache for file handles (np.load with mmap)
#         for file_path in file_list:
#             data = np.load(file_path, mmap_mode='r')
#             n_epochs = data['X'].shape[0]
#             self.epoch_index.extend([(file_path, i) for i in range(n_epochs)])

#     def __len__(self):
#         return len(self.epoch_index)

#     def __getitem__(self, idx):
#         file_path, epoch_idx = self.epoch_index[idx]
#         if file_path not in self.file_data:
#             self.file_data[file_path] = np.load(file_path, mmap_mode='r')
#         X = self.file_data[file_path]['X'][epoch_idx]  # (C, T)
#         Y = self.file_data[file_path]['Y'][epoch_idx]
#         return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)
class EEGDataset(Dataset):
    def __init__(self, file_list):
        self.epoch_paths = []
        for file in file_list:
            data = np.load(file, mmap_mode='r')
            n = len(data['Y'])
            self.epoch_paths.extend([(file, i) for i in range(n)])

    def __len__(self):
        return len(self.epoch_paths)

    def __getitem__(self, idx):
        file, i = self.epoch_paths[idx]
        data = np.load(file, mmap_mode='r')
        X = data['X'][i]   # shape: (C, T)
        Y = data['Y'][i]   # label: 0 or 1
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)



# def collate_fn(batch):
#     Xs, Ys = zip(*batch)
#     X = torch.stack(Xs).unsqueeze(1)  # (B, 1, C, T)
#     Y = torch.stack(Ys)
#     return X, Y
def collate_fn(batch):
    Xs, Ys = zip(*batch)
    X = torch.stack(Xs).unsqueeze(1)  # (B, 1, C, T)
    Y = torch.tensor(Ys)
    return X, Y




def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return acc, f1, auroc, cm


def train_epoch(model, loader, optimizer, loss_fn, device):


    model.train()
    total_loss = 0
    all_preds, all_labels, all_proba = [], [], []
    # for X, Y in loader:
    #     X, Y = X.to(device), Y.to(device)
    #     optimizer.zero_grad()
    #     out = model(X)
    #     proba = nn.functional.softmax(out, dim=1)[:,1]
    #     loss = loss_fn(out, Y)
    #     loss.backward()
    #     optimizer.step()
    #     preds = out.argmax(dim=1)
    #     total_loss += loss.item() * Y.size(0)
    #     all_preds.append(preds.cpu())
    #     all_labels.append(Y.cpu())
    #     all_proba.append(proba.cpu())
    #     if i % 10 == 0:
    #         print(f"Batch {i} Loss: {loss.item():.4f}")
    for X, Y in tqdm(loader, desc="Training", leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        out = model(X)
        proba = nn.functional.softmax(out, dim=1)[:,1]
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=1)
        total_loss += loss.item() * Y.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(Y.cpu())
        all_proba.append(proba.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_proba = torch.cat(all_proba).detach().numpy()
    avg_loss = total_loss / len(y_true)
    
    acc, f1, auroc, cm = compute_metrics(y_true, y_pred, y_proba)
    return avg_loss, acc, f1, auroc, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=['bandpass','gaussian'], default='bandpass')
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir     = os.path.join(project_root, "data", f"epochs_{args.filter}")

    print("Training will load from:", data_dir)
    all_files = glob.glob(os.path.join(data_dir, '*', f"{args.filter}_epochs_*.npz"))
    if args.max_files:
        all_files = all_files[: args.max_files]

    print("Matched files:", all_files)
    if not all_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}. Did you run preprocessing?")    

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_ds = EEGDataset(train_files)
    val_ds = EEGDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=device))

    # for epoch in range(1, args.epochs+1):
    #     train_loss, train_acc, train_auroc, _ = train_epoch(model, train_loader, optimizer, loss_fn, device)
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     val_loss, val_acc, val_auroc, _     = train_epoch(model, val_loader,   optimizer, loss_fn, device)
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     print(f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.4f}, AUROC={train_auroc:.4f} | \
    #           Val loss={val_loss:.4f}, acc={val_acc:.4f}, AUROC={val_auroc:.4f}")
    for epoch in range(1, args.epochs+1):
        # print(f"\nEpoch {epoch}/{args.epochs}")
        # train_loss, train_acc, train_auroc, _ = train_epoch(model, train_loader, optimizer, loss_fn, device, desc="Train")
        # val_loss, val_acc, val_auroc, _     = train_epoch(model, val_loader,   optimizer, loss_fn, device, desc="Val")
        # print(f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, AUROC={train_auroc:.4f} | "
        #       f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUROC={val_auroc:.4f}")
        train_loss, train_acc, train_f1, train_auroc, _ = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc, val_f1, val_auroc, _ = train_epoch(model, val_loader, optimizer, loss_fn, device)

        print(f"Epoch {epoch}: "
            f"Train loss={train_loss:.4f}, acc={train_acc:.4f}, F1={train_f1:.4f}, AUROC={train_auroc:.4f} | "
            f"Val loss={val_loss:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f}, AUROC={val_auroc:.4f}")


    os.makedirs('models/saved', exist_ok=True)
    torch.save(model.state_dict(), f"models/saved/eegnet_{args.filter}.pth")

if __name__ == '__main__':
    main()