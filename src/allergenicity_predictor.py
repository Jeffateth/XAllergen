#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Allergenicity Prediction Model Training Script
Author: Claude (Revised by AI)
Date: May 18, 2025

This script trains and evaluates a Multilayer Perceptron (MLP) model 
for protein allergenicity prediction using ESM-2 embeddings.
It includes grid search hyperparameter optimization, k-fold cross-validation,
and robust measures to prevent data leakage and overfitting.

Usage:
    python allergenicity_predictor.py [--base_dir <path>] [--cv_only] [--test_only] \
                                      [--random_seed <N>] [--n_jobs <N>]

Options:
    --base_dir: Base directory for the project (defaults to script's grandparent directory)
    --cv_only: Only perform cross-validation without training final model
    --test_only: Skip cross-validation and only train/test final model
    --random_seed: Set random seed for reproducibility (default: 42)
    --n_jobs: Number of parallel jobs for grid search (default: -1, use all cores)
"""

import os
from joblib import Parallel, delayed
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import logging
import joblib
import itertools
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*verbose parameter is deprecated.*"
)


# --- Configuration Defaults ---
EMBEDDING_DIM_DEFAULT = 1280  # ESM-2 embedding dimension
BATCH_SIZE_DEFAULT = 32
LEARNING_RATE_DEFAULT = 1e-4
EPOCHS_DEFAULT = 100
PATIENCE_DEFAULT = 15
HIDDEN_LAYERS_DEFAULT = [512, 256, 128]
DROPOUT_RATE_DEFAULT = 0.3
WEIGHT_DECAY_DEFAULT = 1e-4
K_FOLDS_DEFAULT = 5
SEED_DEFAULT = 42
N_JOBS_DEFAULT = -1  # Use all available cores for grid search

logger = logging.getLogger(__name__)

# --- Hyperparameter Grid for Optimization ---
PARAM_GRID = {
    "hidden_layers": [[512, 256, 128], [1024, 512, 256], [512, 256], [256, 128, 64]],
    "learning_rate": [1e-3, 1e-4],
    "dropout_rate": [0.3, 0.4, 0.5],
    "weight_decay": [1e-4, 5e-4, 1e-3],
    "batch_size": [32, 64],
}


# --- Path and Device Setup ---
def get_device():
    """Determines the appropriate device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")  # For Apple Silicon (M1/M2)
    else:
        return torch.device("cpu")


def setup_paths(base_dir_str=None):
    """Set up project paths, creating directories if needed."""
    if base_dir_str:
        try:
            base_dir = Path(base_dir_str).resolve()
            # Verify the provided path exists or can be created
            if not base_dir.exists():
                print(
                    f"Warning: Directory {base_dir} does not exist. Using script location instead."
                )
                base_dir = Path(__file__).resolve().parent.parent
        except Exception as e:
            print(f"Error with provided path: {e}")
            print("Falling back to script location")
            base_dir = Path(__file__).resolve().parent.parent
    else:
        # Default to the script's parent directory
        base_dir = Path(__file__).resolve().parent.parent

    # Define project paths
    paths = {
        "BASE_DIR": base_dir,
        "DATA_DIR": base_dir / "data" / "esm-2-embeddings",
        "MODEL_OUTPUT_DIR": base_dir / "models",
        "RESULTS_DIR": base_dir / "results",
    }

    # Create directories if they don't exist
    for path_name, path in paths.items():
        if isinstance(path, Path):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Couldn't create directory {path}: {e}")
                # If a critical path can't be created, we'll catch errors later when trying to use it

    return paths


def create_run_specific_dirs(paths, timestamp):
    """Creates directories for a specific run, including subdirectories."""
    run_dir = paths["RESULTS_DIR"] / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    dirs = {
        "cv_dir": run_dir / "cross_validation",
        "hyperparam_dir": run_dir / "hyperparameter_search",
        "final_model_dir": run_dir / "final_model_evaluation",
        "model_checkpoints_dir": run_dir / "model_checkpoints",
        "plots_dir": run_dir / "plots",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return run_dir, dirs


# --- Logging and Seeding ---
def setup_logger_config(log_level=logging.INFO, log_file_path=None):
    logger.setLevel(log_level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    if log_file_path:
        f_handler = logging.FileHandler(log_file_path)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)


def set_global_seeds(seed_value):
    """Set seeds for reproducibility across all libraries."""
    logger.info(f"Setting global random seed to {seed_value}")

    # Python's built-in random
    import random

    random.seed(seed_value)

    # NumPy
    np.random.seed(seed_value)

    # PyTorch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed_value)


# --- Dataset and Model ---
class ProteinEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(x))


# --- Data Handling ---
def load_and_preprocess_data(paths, config):
    """Load protein embedding data and perform initial preprocessing.

    Returns unscaled features for proper CV scaling & original shapes for reference.
    """
    logger.info("Loading ESM-2 embeddings data...")
    start_time = time.time()

    train_data_path = paths["DATA_DIR"] / "algpred2_train_esm2_1280dim_embeddings.csv"
    test_data_path = paths["DATA_DIR"] / "algpred2_test_esm2_1280dim_embeddings.csv"

    # Check if data files exist
    if not train_data_path.exists():
        logger.error(f"Training data file not found at: {train_data_path}")
        # Try looking for data in the current directory
        alt_train_path = Path.cwd() / "algpred2_train_esm2_1280dim_embeddings.csv"
        if alt_train_path.exists():
            logger.info(f"Found training data at: {alt_train_path}")
            train_data_path = alt_train_path
        else:
            logger.error("Could not find training data file in alternative locations")
            raise FileNotFoundError(f"Training data file not found")

    if not test_data_path.exists():
        logger.error(f"Test data file not found at: {test_data_path}")
        # Try looking for data in the current directory
        alt_test_path = Path.cwd() / "algpred2_test_esm2_1280dim_embeddings.csv"
        if alt_test_path.exists():
            logger.info(f"Found test data at: {alt_test_path}")
            test_data_path = alt_test_path
        else:
            logger.error("Could not find test data file in alternative locations")
            raise FileNotFoundError(f"Test data file not found")

    # Load data with error handling
    try:
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
    except Exception as e:
        logger.error(f"Error reading data files: {e}")
        raise

    # Extract features and labels
    try:
        # Extract features (columns from index 2 onwards)
        X_train_full = train_df.iloc[:, 2:].values
        # Extract labels (column at index 1)
        y_train_full = train_df.iloc[:, 1].values
        # Extract IDs (column at index 0)
        train_ids_full = train_df.iloc[:, 0].values

        X_test = test_df.iloc[:, 2:].values
        y_test = test_df.iloc[:, 1].values
        test_ids = test_df.iloc[:, 0].values
    except Exception as e:
        logger.error(f"Error extracting features/labels: {e}")
        raise

    # Log data statistics
    logger.info(f"Data loading completed in {time.time() - start_time:.2f}s")
    logger.info(
        f"Full training samples: {len(y_train_full)}, Test samples: {len(y_test)}"
    )
    logger.info(f"Training class distribution: {np.bincount(y_train_full)}")
    logger.info(f"Test class distribution: {np.bincount(y_test)}")

    # Verify embedding dimension
    embedding_dim = X_train_full.shape[1]
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Check for NaN values
    if np.isnan(X_train_full).any() or np.isnan(X_test).any():
        logger.warning("NaN values detected in the data! Replacing with zeros.")
        X_train_full = np.nan_to_num(X_train_full)
        X_test = np.nan_to_num(X_test)

    # Save dataset metadata
    try:
        metadata_path = (
            paths["RESULTS_DIR"]
            / f"run_{config['timestamp']}"
            / "dataset_metadata.json"
        )
        metadata = {
            "train_shape": X_train_full.shape,
            "test_shape": X_test.shape,
            "embedding_dim": embedding_dim,
            "train_label_counts": np.bincount(y_train_full).tolist(),
            "test_label_counts": np.bincount(y_test).tolist(),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        logger.warning(f"Failed to save dataset metadata: {e}")

    # Return unscaled features and reference values
    return (
        X_train_full,
        y_train_full,
        train_ids_full,
        X_test,
        y_test,
        test_ids,
        embedding_dim,
    )


def create_train_val_split(X, y, val_ratio=0.1, random_state=None):
    """Create a train/validation split with stratification."""
    X, y = shuffle(X, y, random_state=random_state)

    # Get indices for each class
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    # Calculate split sizes for each class
    n_val_pos = int(len(pos_indices) * val_ratio)
    n_val_neg = int(len(neg_indices) * val_ratio)

    # Split indices
    val_pos_indices = pos_indices[:n_val_pos]
    train_pos_indices = pos_indices[n_val_pos:]

    val_neg_indices = neg_indices[:n_val_neg]
    train_neg_indices = neg_indices[n_val_neg:]

    # Combine indices
    val_indices = np.concatenate([val_pos_indices, val_neg_indices])
    train_indices = np.concatenate([train_pos_indices, train_neg_indices])

    # Shuffle indices
    np.random.shuffle(val_indices)
    np.random.shuffle(train_indices)

    # Create splits
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    return X_train, y_train, X_val, y_val


# --- Metrics and Plotting ---
def calculate_performance_metrics(y_true, y_pred_scores):
    """Calculate comprehensive performance metrics for binary classification."""
    metrics = {}
    y_pred_binary = (np.array(y_pred_scores) > 0.5).astype(int)
    y_true_np, y_scores_np = np.array(y_true), np.array(y_pred_scores)

    metrics["accuracy"] = accuracy_score(y_true_np, y_pred_binary)
    metrics["precision"] = precision_score(y_true_np, y_pred_binary, zero_division=0)
    metrics["recall"] = recall_score(y_true_np, y_pred_binary, zero_division=0)
    metrics["f1_score"] = f1_score(y_true_np, y_pred_binary, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true_np, y_pred_binary)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Specificity = TN / (TN + FP)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calculate AUC metrics if we have both classes present
    if len(np.unique(y_true_np)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true_np, y_scores_np)
        p_curve, r_curve, _ = precision_recall_curve(y_true_np, y_scores_np)
        metrics["pr_auc"] = auc(r_curve, p_curve)
    else:
        logger.warning(
            "Only one class present in evaluation set! AUC metrics will be invalid."
        )
        metrics["roc_auc"] = 0.5  # Default value
        metrics["pr_auc"] = np.mean(y_true_np)  # Baseline

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def save_plots(
    y_true, y_scores, epoch_num, phase_name, output_dir, calculated_metrics=None
):
    """Generate and save performance plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use passed metrics if available, otherwise calculate
    if calculated_metrics and "roc_auc" in calculated_metrics:
        roc_auc_val = calculated_metrics["roc_auc"]
    else:
        roc_auc_val = roc_auc_score(y_true, y_scores)

    if calculated_metrics and "pr_auc" in calculated_metrics:
        pr_auc_val = calculated_metrics["pr_auc"]
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc_val = auc(recall, precision)

    if calculated_metrics and "confusion_matrix" in calculated_metrics:
        cm_val = np.array(calculated_metrics["confusion_matrix"])
    else:
        cm_val = confusion_matrix(y_true, (np.array(y_scores) > 0.5).astype(int))

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {phase_name} (Epoch {epoch_num})")
    plt.legend()
    plt.savefig(output_dir / f"{phase_name}_roc_epoch_{epoch_num}.png")
    plt.close()

    # Generate PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(10, 8))
    plt.plot(rec, prec, label=f"AUC = {pr_auc_val:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {phase_name} (Epoch {epoch_num})")
    plt.legend()
    plt.savefig(output_dir / f"{phase_name}_pr_epoch_{epoch_num}.png")
    plt.close()

    # Generate confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {phase_name} (Epoch {epoch_num})")
    plt.savefig(output_dir / f"{phase_name}_cm_epoch_{epoch_num}.png")
    plt.close()


def plot_aggregate_training_history(history_df, output_dir, prefix=""):
    """Generate a comprehensive training history plot."""
    if history_df.empty:
        logger.warning("Empty history dataframe, skipping training history plot")
        return

    plt.figure(figsize=(18, 12))

    metrics_to_plot = ["loss", "roc_auc", "accuracy", "f1_score", "pr_auc", "mcc"]
    for i, metric in enumerate(metrics_to_plot):
        # Skip metrics not found in history_df
        if not any(col.endswith(metric) for col in history_df.columns):
            continue

        plt.subplot(2, 3, (i % 6) + 1)

        # Plot train metric if available
        if f"train_{metric}" in history_df.columns:
            plt.plot(
                history_df["epoch"],
                history_df[f"train_{metric}"],
                label=f'Train {metric.replace("_"," ").title()}',
            )

        # Plot validation metric if available
        if f"val_{metric}" in history_df.columns:
            plt.plot(
                history_df["epoch"],
                history_df[f"val_{metric}"],
                label=f'Val {metric.replace("_"," ").title()}',
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f'{metric.replace("_"," ").title()} vs. Epoch')
        plt.legend()

    # Plot learning rate if available
    if (
        "learning_rate" in history_df.columns
        and history_df["learning_rate"].nunique() > 1
    ):
        plt.subplot(2, 3, 6)
        plt.plot(
            history_df["epoch"], history_df["learning_rate"], label="Learning Rate"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate vs. Epoch")
        plt.legend()
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}training_history.png")
    plt.close()


# --- Training and Evaluation Loop ---
def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    """Run a single training or evaluation epoch."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device).unsqueeze(1)

            if is_train:
                optimizer.zero_grad()

            outputs = model(embeddings)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * embeddings.size(0)
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    metrics = calculate_performance_metrics(all_targets, all_preds)
    return avg_loss, metrics, all_targets, all_preds


def train_and_validate_model(
    model_config,
    data_loaders,
    device,
    output_dir,
    checkpoint_base_name="model",
    resume_path=None,
):
    """Train a model with given data loaders, supports resuming training."""
    train_loader, val_loader = data_loaders["train"], data_loaders["val"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MLPClassifier(
        input_dim=model_config["input_dim"],
        hidden_dims=model_config["hidden_layers"],
        dropout_rate=model_config["dropout_rate"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
    )

    criterion = nn.BCELoss()

    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=model_config.get("lr_patience", 5),
        verbose=True,
    )

    # Initialize training state
    start_epoch = 0
    best_val_metrics = {"roc_auc": 0.0, "pr_auc": 0.0, "mcc": -1.0, "f1_score": 0.0}
    patience_counter = 0
    history = []

    # Set paths for model checkpoints
    checkpoint_path = output_dir / f"{checkpoint_base_name}_checkpoint.pt"

    # Resume training if requested
    if resume_path and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device)
        logger.info(f"Resuming training from explicit path: {resume_path}")
    elif model_config.get("resume", False) and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Resuming training from default checkpoint: {checkpoint_path}")
    else:
        checkpoint = None

    # Load state if resuming
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_metrics = checkpoint.get("best_val_metrics", best_val_metrics)
        history = checkpoint.get("history", [])
        logger.info(
            f"Resumed from epoch {start_epoch}. Best Val ROC AUC: {best_val_metrics['roc_auc']:.4f}"
        )

    # Main training loop
    for epoch in range(start_epoch, model_config["epochs"]):
        epoch_time = time.time()

        # Training phase
        train_loss, train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )

        # Validation phase
        val_loss, val_metrics, val_targets, val_scores = run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )

        # Update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["roc_auc"])

        # Compile epoch summary
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_roc_auc": train_metrics["roc_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_f1": train_metrics["f1_score"],
            "val_f1": val_metrics["f1_score"],
            "train_mcc": train_metrics["mcc"],
            "val_mcc": val_metrics["mcc"],
            "learning_rate": current_lr,
            "epoch_time_s": time.time() - epoch_time,
        }
        history.append(epoch_summary)

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_metrics": best_val_metrics,
                "history": history,
            },
            checkpoint_path,
        )

        # Check for new best model based on primary metric (ROC AUC)
        primary_metric_improved = val_metrics["roc_auc"] > best_val_metrics["roc_auc"]

        # Update best metrics
        metrics_improved = {}
        for metric_name in best_val_metrics:
            if metric_name in ["roc_auc", "pr_auc", "f1_score", "accuracy"]:
                # Higher is better
                if val_metrics[metric_name] > best_val_metrics[metric_name]:
                    metrics_improved[metric_name] = True
                    best_val_metrics[metric_name] = val_metrics[metric_name]
            elif metric_name in ["mcc"]:
                # Higher is better for MCC too
                if val_metrics[metric_name] > best_val_metrics[metric_name]:
                    metrics_improved[metric_name] = True
                    best_val_metrics[metric_name] = val_metrics[metric_name]

        # Save best models for each metric
        if primary_metric_improved:
            torch.save(
                model.state_dict(), output_dir / f"{checkpoint_base_name}_best_auc.pt"
            )
            logger.info(
                f"Epoch {epoch+1}: New best Val AUC: {best_val_metrics['roc_auc']:.4f}. Model saved."
            )
            save_plots(
                val_targets,
                val_scores,
                epoch + 1,
                "validation_best",
                output_dir,
                val_metrics,
            )
            patience_counter = 0
        else:
            patience_counter += 1

        # For other metrics that improved
        for metric_name, improved in metrics_improved.items():
            if (
                improved and metric_name != "roc_auc"
            ):  # Already handled the primary metric
                torch.save(
                    model.state_dict(),
                    output_dir / f"{checkpoint_base_name}_best_{metric_name}.pt",
                )
                logger.debug(
                    f"Saved new best model for {metric_name}: {best_val_metrics[metric_name]:.4f}"
                )

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{model_config['epochs']} | "
            f"Train Loss: {train_loss:.4f} AUC: {train_metrics['roc_auc']:.4f} | "
            f"Val Loss: {val_loss:.4f} AUC: {val_metrics['roc_auc']:.4f} | "
            f"Patience: {patience_counter}/{model_config.get('patience', 15)}"
        )

        # Early stopping check
        if patience_counter >= model_config.get("patience", 15):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Convert training history to DataFrame
    history_df = pd.DataFrame(history)
    return model, history_df, best_val_metrics


from tqdm import trange
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def run_optuna_search(
    X, y, embedding_dim, device, random_seed, output_dir, n_trials=25, k_folds=3
):
    logger.info(f"Starting Optuna Hyperparameter Optimization with {n_trials} trials")

    def objective(trial):
        hidden_layer_size = trial.suggest_categorical(
            "hidden_layers",
            [
                [512, 256],
                [256, 128],
                [512, 256, 128],
            ],
        )
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])

        kfold = StratifiedKFold(
            n_splits=k_folds, shuffle=True, random_state=random_seed
        )
        val_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            train_loader = DataLoader(
                ProteinEmbeddingDataset(X_train_scaled, y_train),
                batch_size=batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                ProteinEmbeddingDataset(X_val_scaled, y_val),
                batch_size=batch_size,
                shuffle=False,
            )

            model_config = {
                "input_dim": embedding_dim,
                "hidden_layers": hidden_layer_size,
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": 10,  # keep small for speed
                "patience": 3,
                "lr_patience": 2,
                "resume": False,
            }

            fold_output_dir = (
                output_dir / f"optuna_trial_{trial.number}_fold_{fold_idx}"
            )
            fold_output_dir.mkdir(parents=True, exist_ok=True)

            model, _, val_metrics = train_and_validate_model(
                model_config=model_config,
                data_loaders={"train": train_loader, "val": val_loader},
                device=device,
                output_dir=fold_output_dir,
                checkpoint_base_name="optuna_model",
            )

            val_scores.append(val_metrics["roc_auc"])

        return np.mean(val_scores)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed)
    )

    # Manual loop with progress bar
    for _ in trange(n_trials, desc="Optuna Trials"):
        study.optimize(objective, n_trials=1, n_jobs=1)

    logger.info(f"Best Trial: {study.best_trial}")
    return study.best_trial.params


from joblib import Parallel, delayed


def run_grid_search_cv(
    X,
    y,
    param_grid,
    embedding_dim,
    device,
    random_seed,
    output_dir,
    k_folds=5,
    n_jobs=-1,
):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    import json

    logger.info("Starting Parallel Grid Search with Cross-Validation")
    all_params = list(ParameterGrid(param_grid))

    # STEP 1: Ensure all temp directories exist before parallel execution
    for i in range(len(all_params)):
        temp_dir = output_dir / f"gridsearch_temp_set_{i}"
        temp_dir.mkdir(parents=True, exist_ok=True)

    # STEP 2: Define the worker function to evaluate a single hyperparameter set
    def evaluate_params(i, params):
        val_scores = []
        kfold = StratifiedKFold(
            n_splits=k_folds, shuffle=True, random_state=random_seed
        )

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            train_loader = DataLoader(
                ProteinEmbeddingDataset(X_train_scaled, y_train),
                batch_size=params["batch_size"],
                shuffle=True,
            )
            val_loader = DataLoader(
                ProteinEmbeddingDataset(X_val_scaled, y_val),
                batch_size=params["batch_size"],
                shuffle=False,
            )

            model_config = {
                "input_dim": embedding_dim,
                "hidden_layers": params["hidden_layers"],
                "learning_rate": params["learning_rate"],
                "dropout_rate": params["dropout_rate"],
                "weight_decay": params["weight_decay"],
                "batch_size": params["batch_size"],
                "epochs": 50,
                "patience": 7,
                "lr_patience": 3,
                "resume": False,
            }

            model, _, val_metrics = train_and_validate_model(
                model_config=model_config,
                data_loaders={"train": train_loader, "val": val_loader},
                device=device,
                output_dir=output_dir / f"gridsearch_temp_set_{i}",
                checkpoint_base_name=f"grid_fold_{fold_idx}_set_{i}",
            )

            val_scores.append(val_metrics["roc_auc"])

        mean_score = np.mean(val_scores)
        return (params, mean_score)

    # STEP 3: Run all parameter sets in parallel
    from tqdm import tqdm

    results = Parallel(n_jobs=3)(
        delayed(evaluate_params)(i, params)
        for i, params in tqdm(list(enumerate(all_params)), desc="Grid Search Progress")
    )

    # STEP 4: Log and save results
    best_config = max(results, key=lambda x: x[1])[0]
    best_score = max(results, key=lambda x: x[1])[1]

    logger.info(f"Best configuration: {best_config} with ROC AUC = {best_score:.4f}")

    pd.DataFrame(
        [{"params": json.dumps(p), "mean_roc_auc": s} for p, s in results]
    ).to_csv(output_dir / "grid_search_results.csv", index=False)

    return best_config


# --- Final Model Training & Evaluation ---
def train_and_evaluate_final_model(
    X_train_all,
    y_train_all,
    X_test,
    y_test,
    test_ids,
    model_config,
    device,
    final_model_dir,
    model_checkpoints_dir,
    random_seed,
    resume=False,
):
    """Train final model on all training data and evaluate on test set."""
    logger.info("\n--- Training Final Model on Full Training Data ---")

    # Scale features using entire training set (no data leakage as test set is separate)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for deployment
    scaler_path = final_model_dir / "final_model_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Final model scaler saved to {scaler_path}")

    # Create a train/validation split from the full training data for the final model
    X_train_final, y_train_final, X_val_final, y_val_final = create_train_val_split(
        X_train_scaled, y_train_all, val_ratio=0.1, random_state=random_seed
    )

    # Create datasets
    train_dataset = ProteinEmbeddingDataset(X_train_final, y_train_final)
    val_dataset = ProteinEmbeddingDataset(X_val_final, y_val_final)
    test_dataset = ProteinEmbeddingDataset(X_test_scaled, y_test)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=model_config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=model_config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=model_config["batch_size"], shuffle=False
    )

    # Data loaders for model training
    final_model_data_loaders = {"train": train_loader, "val": val_loader}

    # Path for resuming final model if requested
    final_model_resume_path = (
        model_checkpoints_dir / "final_model_checkpoint.pt" if resume else None
    )

    # Train final model
    final_model, final_history_df, _ = train_and_validate_model(
        model_config=model_config,
        data_loaders=final_model_data_loaders,
        device=device,
        output_dir=model_checkpoints_dir,
        checkpoint_base_name="final_model",
        resume_path=final_model_resume_path,
    )

    # Save training history
    if not final_history_df.empty:
        plot_aggregate_training_history(
            final_history_df, final_model_dir, prefix="final_model_"
        )
        final_history_df.to_csv(
            final_model_dir / "final_model_training_history.csv", index=False
        )

    # Evaluate final model on test set
    logger.info("\n--- Evaluating Final Model on Test Set ---")

    # Load the best performing version of the final model for testing
    best_final_model_path = model_checkpoints_dir / "final_model_best_auc.pt"
    if best_final_model_path.exists():
        final_model.load_state_dict(
            torch.load(best_final_model_path, map_location=device)
        )
        logger.info(f"Loaded best final model from: {best_final_model_path}")
    else:
        logger.warning(
            "Best final model not found. Evaluating with the last trained state."
        )

    # Run evaluation on test set
    test_loss, test_metrics, test_targets, test_scores = run_epoch(
        model=final_model,
        loader=test_loader,
        criterion=nn.BCELoss(),
        optimizer=None,  # No optimization during evaluation
        device=device,
        is_train=False,
    )

    # Log test performance
    logger.info(f"Final Test Metrics - Loss: {test_loss:.4f}")
    for metric, value in test_metrics.items():
        if metric != "confusion_matrix":  # Skip confusion matrix in logs
            logger.info(f"  {metric}: {value:.4f}")

    # Save test performance plots
    save_plots(
        y_true=test_targets,
        y_scores=test_scores,
        epoch_num="final",
        phase_name="test",
        output_dir=final_model_dir,
        calculated_metrics=test_metrics,
    )

    # Save detailed test predictions
    test_results_df = pd.DataFrame(
        {
            "protein_id": test_ids[: len(test_scores)],
            "true_label": test_targets,
            "predicted_score": test_scores,
            "predicted_class": (np.array(test_scores) > 0.5).astype(int),
        }
    )
    test_results_df.to_csv(
        final_model_dir / "final_model_test_predictions.csv", index=False
    )

    # Save test metrics for reference
    with open(final_model_dir / "final_model_test_metrics.json", "w") as f:
        # Convert numpy types for JSON serialization
        serializable_metrics = {}
        for k, v in test_metrics.items():
            if isinstance(v, (np.ndarray, list)):
                if isinstance(v, np.ndarray):
                    serializable_metrics[k] = v.tolist()
                else:
                    serializable_metrics[k] = v
            elif isinstance(v, (np.float32, np.float64)):
                serializable_metrics[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                serializable_metrics[k] = int(v)
            else:
                serializable_metrics[k] = v
        json.dump(serializable_metrics, f, indent=4)

    logger.info(f"Saved final test metrics and predictions to {final_model_dir}")


def run_pipeline(base_dir, cv_only, test_only, random_seed, n_jobs):

    # Timestamp for output dirs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Paths & logging
    paths = setup_paths(base_dir)
    run_dir, dirs = create_run_specific_dirs(paths, timestamp)
    setup_logger_config(log_file_path=run_dir / "training.log")
    set_global_seeds(random_seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Config
    config = {
        "timestamp": timestamp,
        "random_seed": random_seed,
        "n_jobs": n_jobs,
        "device": device,
    }

    # Load data
    (
        X_train_full,
        y_train_full,
        train_ids_full,
        X_test,
        y_test,
        test_ids,
        embedding_dim,
    ) = load_and_preprocess_data(paths, config)

    # Define best model config (replace with grid search result if needed)
    best_model_config = {
        "input_dim": embedding_dim,
        "hidden_layers": HIDDEN_LAYERS_DEFAULT,
        "learning_rate": LEARNING_RATE_DEFAULT,
        "dropout_rate": DROPOUT_RATE_DEFAULT,
        "weight_decay": WEIGHT_DECAY_DEFAULT,
        "batch_size": BATCH_SIZE_DEFAULT,
        "epochs": EPOCHS_DEFAULT,
        "patience": PATIENCE_DEFAULT,
        "lr_patience": 5,
        "resume": False,  # You can change this if needed
    }

    # Cross-validation
    if not test_only:
        logger.info("--- Starting Cross-Validation (skipped in this snippet) ---")
        # You can plug in your CV code here using StratifiedKFold and the above functions

        # Run Grid Search to find best hyperparameters
    if not test_only:
        best_hyperparams = run_optuna_search(
            X=X_train_full,
            y=y_train_full,
            embedding_dim=embedding_dim,
            device=device,
            random_seed=random_seed,
            output_dir=dirs["hyperparam_dir"],
            n_trials=30,  # tune this
        )

        # Update config with best hyperparameters
        best_model_config.update(best_hyperparams)
    # Final training
    if not cv_only:
        train_and_evaluate_final_model(
            X_train_all=X_train_full,
            y_train_all=y_train_full,
            X_test=X_test,
            y_test=y_test,
            test_ids=test_ids,
            model_config=best_model_config,
            device=device,
            final_model_dir=dirs["final_model_dir"],
            model_checkpoints_dir=dirs["model_checkpoints_dir"],
            random_seed=random_seed,
            resume=False,
        )


# --- Main Function ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Allergenicity Predictor MLP")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--cv_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()

    run_pipeline(
        base_dir=args.base_dir,
        cv_only=args.cv_only,
        test_only=args.test_only,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )
