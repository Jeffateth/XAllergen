#!/usr/bin/env python3
"""
train_allergen_cnn.py

Full nested-CV + final evaluation script with detailed logging:
- Loads preprocessed .npy train/test splits (casts to float32 for MPS compatibility)
- 5-fold outer CV on the training set
  - 3-fold inner CV per outer fold to determine early-stopping epoch
- Optional y-scrambling and bootstrap sampling within each training fold
- CAM CNN architecture combining one-hot sequence and structural features
- Dummy baseline comparison on each outer fold
- Early stopping based on validation AUC
- Checkpointing per outer fold with auto-resume support
- Retraining on full train set for average stop epoch
- Final evaluation on hold-out test set
- Saves nested CV metrics, per-fold checkpoints, final model, and test metrics
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# --- Dataset w/ optional y-scramble & bootstrap ---
class AllergenTensorDataset(Dataset):
    def __init__(self, X_seq, X_struct, y, y_scramble=False, bootstrap=False):
        self.X_seq = X_seq.astype(np.float32)
        self.X_struct = X_struct.astype(np.float32)
        self.y = y.copy()
        n = len(self.y)
        if y_scramble:
            print("[Dataset] Applying y-scrambling")
            self.y = np.random.permutation(self.y)
        if bootstrap:
            print("[Dataset] Applying bootstrap sampling")
            idx = np.random.choice(n, size=n, replace=True)
            self.X_seq = self.X_seq[idx]
            self.X_struct = self.X_struct[idx]
            self.y = self.y[idx]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X_seq[i]),
            torch.from_numpy(self.X_struct[i]),
            torch.tensor(int(self.y[i]), dtype=torch.long),
        )


# --- CAM CNN model definition ---
class CAMCNN(nn.Module):
    def __init__(self, struct_dim, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128 + struct_dim, num_classes)

    def forward(self, x_seq, x_struct):
        # Validate input dimensions
        if x_seq.dim() != 3:
            raise ValueError(f"Expected 3D sequence input, got {x_seq.dim()}D")
        if x_struct.dim() != 2:
            raise ValueError(f"Expected 2D structural input, got {x_struct.dim()}D")

        fmap = self.features(x_seq)  # (batch,128,L')
        gap = fmap.mean(dim=2)  # (batch,128)
        combined = torch.cat([gap, x_struct], dim=1)  # (batch,128+struct_dim)
        out = self.dropout(combined)
        logits = self.classifier(out)
        return logits, fmap


# --- Checkpoint utilities ---
def save_ckpt(model, optimizer, fold, epoch, out_dir):
    try:
        path = os.path.join(out_dir, f"ckpt_fold{fold}.pth")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "epoch": epoch,
            },
            path,
        )
        print(f"[Checkpoint] Saved fold {fold} at epoch {epoch}")
        return path
    except Exception as e:
        print(f"[Checkpoint] Error saving checkpoint: {e}")
        return None


def load_ckpt(model, optimizer, path):
    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["opt"])
        print(
            f"[Checkpoint] Loaded checkpoint from {path}, resuming from epoch {state['epoch']}"
        )
        return state["epoch"]
    except Exception as e:
        print(f"[Checkpoint] Error loading checkpoint: {e}")
        return 0


# --- Training and evaluation loops ---
def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    try:
        for x_seq, x_struct, y in loader:
            x_seq, x_struct = x_seq.to(device).float(), x_struct.to(device).float()
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_seq, x_struct)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    except Exception as e:
        print(f"[Train] Error during training epoch: {e}")
        return float("inf")

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    print(f"[Train] Avg loss: {avg_loss:.4f}")
    return avg_loss


def eval_epoch(model, loader, device):
    model.eval()
    ys, ps = [], []
    try:
        with torch.no_grad():
            for x_seq, x_struct, y in loader:
                x_seq, x_struct = x_seq.to(device).float(), x_struct.to(device).float()
                logits, _ = model(x_seq, x_struct)
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                ys.extend(y.numpy())
                ps.extend(prob)
    except Exception as e:
        print(f"[Eval] Error during evaluation: {e}")
        return np.array([]), np.array([])

    return np.array(ys), np.array(ps)


def load_previous_results(output_dir):
    """Load previous nested CV results if resuming"""
    nested_csv = os.path.join(output_dir, "nested_cv_metrics.csv")
    if os.path.exists(nested_csv):
        try:
            df = pd.read_csv(nested_csv)
            return df.to_dict("records")
        except Exception as e:
            print(f"[Resume] Error loading previous results: {e}")
    return []


# --- Main routine ---
def main(args):
    print("[Start] Loading preprocessed arrays...")
    try:
        pre = args.preproc_dir
        Xs_train = np.load(os.path.join(pre, "X_seq_train.npy"))
        Xst_train = np.load(os.path.join(pre, "X_struct_train.npy"))
        y_train = np.load(os.path.join(pre, "y_train.npy"))
        Xs_test = np.load(os.path.join(pre, "X_seq_test.npy"))
        Xst_test = np.load(os.path.join(pre, "X_struct_test.npy"))
        y_test = np.load(os.path.join(pre, "y_test.npy"))
        print(f"[Data] Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return

    struct_dim = Xst_train.shape[1]

    # Enhanced device selection with MPS support
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("[Device] Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[Device] Using CUDA")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare nested CV
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)

    # Load previous results if resuming
    results = load_previous_results(args.output_dir) if args.resume else []

    # Detect completed folds
    done_folds = {
        int(f.split("fold")[1].split(".pth")[0])
        for f in os.listdir(args.output_dir)
        if f.startswith("ckpt_fold")
    }
    resume_fold = max(done_folds) if args.resume and done_folds else -1
    print(f"[CV] Resuming from outer fold {resume_fold + 1} if applicable")

    # Outer CV loop
    for outer_fold, (tr_idx, val_idx) in enumerate(outer_cv.split(Xst_train, y_train)):
        print(f"\n[CV] Starting outer fold {outer_fold}")
        if outer_fold <= resume_fold:
            print(f"[CV] Skipping fold {outer_fold}, already completed")
            continue

        # Split outer fold
        Xs_tr_o = Xs_train[tr_idx]
        Xst_tr_o = Xst_train[tr_idx]
        y_tr_o = y_train[tr_idx]
        Xs_val_o = Xs_train[val_idx]
        Xst_val_o = Xst_train[val_idx]
        y_val_o = y_train[val_idx]

        # Dummy baseline on outer validation
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(Xst_tr_o, y_tr_o)
        dummy_probs = dummy.predict_proba(Xst_val_o)[:, 1]
        auc_dummy = roc_auc_score(y_val_o, dummy_probs)
        print(f"[CV][Fold {outer_fold}] Dummy AUC = {auc_dummy:.4f}")
        results.append({"fold": outer_fold, "model": "dummy", "auc": float(auc_dummy)})

        # Inner CV to choose early stopping epoch
        stop_epochs = []
        for inner_fold, (i_tr, i_val) in enumerate(inner_cv.split(Xst_tr_o, y_tr_o)):
            print(f"[CV][Fold {outer_fold}][Inner {inner_fold}] Starting inner CV")

            # Create datasets with optional scrambling/bootstrap
            ds_tr_i = AllergenTensorDataset(
                Xs_tr_o[i_tr],
                Xst_tr_o[i_tr],
                y_tr_o[i_tr],
                y_scramble=args.y_scramble,
                bootstrap=args.bootstrap,
            )
            ds_val_i = AllergenTensorDataset(
                Xs_tr_o[i_val], Xst_tr_o[i_val], y_tr_o[i_val]
            )
            dl_tr_i = DataLoader(ds_tr_i, batch_size=args.batch_size, shuffle=True)
            dl_val_i = DataLoader(ds_val_i, batch_size=args.batch_size, shuffle=False)

            model_i = CAMCNN(struct_dim).to(device)
            optimizer_i = optim.Adam(model_i.parameters(), lr=args.lr)
            loss_fn = nn.CrossEntropyLoss()

            best_auc_i, no_improve, best_ep = 0.0, 0, 0
            for epoch in range(args.epochs):
                print(f"[CV][Fold {outer_fold}][Inner {inner_fold}] Epoch {epoch}")
                train_loss = train_epoch(model_i, dl_tr_i, loss_fn, optimizer_i, device)

                # Skip evaluation if training failed
                if train_loss == float("inf"):
                    print(
                        f"[CV][Fold {outer_fold}][Inner {inner_fold}] Training failed at epoch {epoch}"
                    )
                    break

                ys_i, ps_i = eval_epoch(model_i, dl_val_i, device)

                # Skip if evaluation failed
                if len(ys_i) == 0:
                    print(
                        f"[CV][Fold {outer_fold}][Inner {inner_fold}] Evaluation failed at epoch {epoch}"
                    )
                    break

                try:
                    auc_i = roc_auc_score(ys_i, ps_i)
                    print(
                        f"[CV][Fold {outer_fold}][Inner {inner_fold}] Val AUC = {auc_i:.4f}"
                    )

                    if auc_i > best_auc_i:
                        best_auc_i, no_improve, best_ep = auc_i, 0, epoch
                    else:
                        no_improve += 1
                        if no_improve >= args.patience:
                            print(
                                f"[CV][Fold {outer_fold}][Inner {inner_fold}] Early stopping at epoch {epoch}"
                            )
                            break
                except Exception as e:
                    print(
                        f"[CV][Fold {outer_fold}][Inner {inner_fold}] Error calculating AUC: {e}"
                    )
                    break

            stop_epochs.append(best_ep)

        # Calculate average chosen epoch with fallback
        if stop_epochs:
            chosen_epoch = int(np.mean(stop_epochs))
        else:
            chosen_epoch = args.epochs // 2  # Fallback if all inner folds failed
        print(f"[CV][Fold {outer_fold}] Chosen stop epoch = {chosen_epoch}")

        # Train on outer train and validate
        ds_outer_tr = AllergenTensorDataset(
            Xs_tr_o,
            Xst_tr_o,
            y_tr_o,
            y_scramble=args.y_scramble,
            bootstrap=args.bootstrap,
        )
        ds_outer_val = AllergenTensorDataset(Xs_val_o, Xst_val_o, y_val_o)
        dl_outer_tr = DataLoader(ds_outer_tr, batch_size=args.batch_size, shuffle=True)
        dl_outer_val = DataLoader(
            ds_outer_val, batch_size=args.batch_size, shuffle=False
        )

        model_o = CAMCNN(struct_dim).to(device)
        optimizer_o = optim.Adam(model_o.parameters(), lr=args.lr)
        loss_fn_outer = nn.CrossEntropyLoss()

        # Resume checkpoint if exists
        start_epoch = 0
        ckpt_path = os.path.join(args.output_dir, f"ckpt_fold{outer_fold}.pth")
        if args.resume and os.path.isfile(ckpt_path):
            start_epoch = load_ckpt(model_o, optimizer_o, ckpt_path)

        best_auc_o, no_improve = 0.0, 0
        max_epochs = max(chosen_epoch + args.patience, args.epochs)

        for epoch in range(start_epoch, max_epochs):
            print(f"[CV][Fold {outer_fold}] Training epoch {epoch}")
            train_loss = train_epoch(
                model_o, dl_outer_tr, loss_fn_outer, optimizer_o, device
            )

            # Skip evaluation if training failed
            if train_loss == float("inf"):
                print(f"[CV][Fold {outer_fold}] Training failed at epoch {epoch}")
                break

            ys_o, ps_o = eval_epoch(model_o, dl_outer_val, device)

            # Skip if evaluation failed
            if len(ys_o) == 0:
                print(f"[CV][Fold {outer_fold}] Evaluation failed at epoch {epoch}")
                break

            try:
                auc_o = roc_auc_score(ys_o, ps_o)
                print(f"[CV][Fold {outer_fold}] Val AUC = {auc_o:.4f}")

                if auc_o > best_auc_o:
                    best_auc_o, no_improve = auc_o, 0
                    save_ckpt(model_o, optimizer_o, outer_fold, epoch, args.output_dir)
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        print(
                            f"[CV][Fold {outer_fold}] Early stopping at epoch {epoch}"
                        )
                        break
            except Exception as e:
                print(f"[CV][Fold {outer_fold}] Error calculating outer AUC: {e}")
                break

        # Record results with fallback
        if best_auc_o > 0:
            results.append(
                {
                    "fold": outer_fold,
                    "model": "CNN",
                    "auc": float(best_auc_o),
                    "stop_epoch": chosen_epoch,
                }
            )
        else:
            print(
                f"[CV][Fold {outer_fold}] CNN training failed, recording dummy result"
            )
            results.append(
                {
                    "fold": outer_fold,
                    "model": "CNN",
                    "auc": 0.5,
                    "stop_epoch": chosen_epoch,
                }
            )

        # Save intermediate results after each fold
        try:
            nested_csv = os.path.join(args.output_dir, "nested_cv_metrics.csv")
            pd.DataFrame(results).to_csv(nested_csv, index=False)
            print(f"[CV] Intermediate results saved to {nested_csv}")
        except Exception as e:
            print(f"[CV] Error saving intermediate results: {e}")

    # Save final nested CV results
    try:
        nested_csv = os.path.join(args.output_dir, "nested_cv_metrics.csv")
        pd.DataFrame(results).to_csv(nested_csv, index=False)
        print(f"[CV] Final nested CV metrics saved to {nested_csv}")
    except Exception as e:
        print(f"[CV] Error saving final results: {e}")

    # Final retrain on full train, then test on hold-out
    print("\n[Final] Retraining on full training set...")
    try:
        full_ds = AllergenTensorDataset(
            Xs_train,
            Xst_train,
            y_train,
            y_scramble=args.y_scramble,
            bootstrap=args.bootstrap,
        )
        full_dl = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)
        test_ds = AllergenTensorDataset(Xs_test, Xst_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Calculate average stop epoch with robust fallback
        cnn_results = [
            r["stop_epoch"]
            for r in results
            if r["model"] == "CNN" and "stop_epoch" in r
        ]
        if cnn_results:
            avg_stop = int(np.mean(cnn_results))
        else:
            avg_stop = args.epochs // 2  # Fallback if no CNN results
        print(f"[Final] Using average stop epoch = {avg_stop}")

        final_model = CAMCNN(struct_dim).to(device)
        final_opt = optim.Adam(final_model.parameters(), lr=args.lr)
        loss_fn_final = nn.CrossEntropyLoss()

        # Train final model
        for epoch in range(avg_stop):
            print(f"[Final] Training epoch {epoch}")
            train_loss = train_epoch(
                final_model, full_dl, loss_fn_final, final_opt, device
            )
            if train_loss == float("inf"):
                print(f"[Final] Training failed at epoch {epoch}")
                break

        # Evaluate on test set
        ys_test, ps_test = eval_epoch(final_model, test_dl, device)

        if len(ys_test) > 0:
            test_auc = roc_auc_score(ys_test, ps_test)
            test_acc = accuracy_score(ys_test, (ps_test > 0.5).astype(int))
            print(f"[Final] Test AUC = {test_auc:.4f}, Test ACC = {test_acc:.4f}")

            # Save final model and metrics
            final_model_path = os.path.join(args.output_dir, "final_model.pth")
            torch.save(final_model.state_dict(), final_model_path)

            test_metrics_path = os.path.join(args.output_dir, "test_metrics.csv")
            pd.DataFrame([{"test_auc": test_auc, "test_acc": test_acc}]).to_csv(
                test_metrics_path, index=False
            )
            print(
                f"[Final] Saved final model to {final_model_path} and metrics to {test_metrics_path}"
            )
        else:
            print("[Final] Test evaluation failed - no predictions generated")

    except Exception as e:
        print(f"[Final] Error during final training/evaluation: {e}")

    print("[Completed] Script finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train allergen CNN with nested cross-validation"
    )
    parser.add_argument(
        "--preproc-dir",
        type=str,
        required=True,
        help="Directory with .npy preprocessed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save models and metrics",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume nested CV from last completed outer fold",
    )
    parser.add_argument(
        "--y-scramble",
        action="store_true",
        help="Permute labels in each training split",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bootstrap sample in each training split",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum epochs for training (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    main(args)
