#!/usr/bin/env python3
"""
evaluate_allergen_model.py

Comprehensive evaluation script for allergen models:
- Loads model checkpoints (either single fold or final model)
- Generates predictions on test data
- Creates detailed visualizations (ROC curve, confusion matrix)
- Outputs comprehensive classification metrics
- Saves all visualizations and metrics to outputs directory
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)

# Import the model architecture from training script
from train_allergen_cnn import CAMCNN, AllergenTensorDataset
from torch.utils.data import DataLoader


def setup_device():
    """Set up and return the appropriate device for computation"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("[Device] Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[Device] Using CUDA")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


def load_model(model_path, struct_dim, device):
    """Load a trained model from checkpoint"""
    model = CAMCNN(struct_dim).to(device)
    try:
        # Try loading as full checkpoint (including optimizer state)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            print(f"[Model] Loaded from checkpoint: {model_path}")
        else:
            # Direct model state dict
            model.load_state_dict(checkpoint)
            print(f"[Model] Loaded state dict: {model_path}")
        return model
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return None


def load_data(data_dir):
    """Load preprocessed test data"""
    try:
        X_seq_test = np.load(os.path.join(data_dir, "X_seq_test.npy"))
        X_struct_test = np.load(os.path.join(data_dir, "X_struct_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))
        ids_test = np.load(os.path.join(data_dir, "ids_test.npy"))

        print(f"[Data] Loaded test data: {len(y_test)} samples")
        print(f"[Data] X_seq shape: {X_seq_test.shape}")
        print(f"[Data] X_struct shape: {X_struct_test.shape}")

        # Verify positive/negative distribution
        pos_count = np.sum(y_test)
        print(
            f"[Data] Class distribution: {pos_count} allergens, {len(y_test) - pos_count} non-allergens"
        )

        return X_seq_test, X_struct_test, y_test, ids_test
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return None, None, None, None


def get_predictions(model, X_seq, X_struct, y_true, device, batch_size=32):
    """Get model predictions on test data"""
    dataset = AllergenTensorDataset(X_seq, X_struct, y_true)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    all_labels = []
    all_feature_maps = []

    with torch.no_grad():
        for x_seq, x_struct, y in dataloader:
            x_seq = x_seq.to(device).float()
            x_struct = x_struct.to(device).float()
            y = y.to(device)

            logits, feature_maps = model(x_seq, x_struct)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())

            # Optional: save feature maps for visualization
            if feature_maps is not None:
                all_feature_maps.append(feature_maps.cpu().numpy())

    return np.array(all_labels), np.array(all_probs), all_feature_maps


def plot_roc_curve(y_true, y_pred_probs, output_dir):
    """Create and save ROC curve plot"""
    plt.figure(figsize=(10, 8))

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    auc = roc_auc_score(y_true, y_pred_probs)

    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save plot
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] ROC curve saved to {roc_path}")

    # Save ROC data for further analysis
    roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    roc_data_path = os.path.join(output_dir, "roc_data.csv")
    roc_data.to_csv(roc_data_path, index=False)

    return auc


def plot_confusion_matrix(y_true, y_pred_probs, threshold=0.5, output_dir=None):
    """Create and save confusion matrix visualization"""
    y_pred = (y_pred_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # Using Seaborn for better visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Allergen", "Allergen"],
        yticklabels=["Non-Allergen", "Allergen"],
    )
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.title(f"Confusion Matrix (Threshold = {threshold:.2f})", fontsize=16)

    # Save plot
    if output_dir:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        print(f"[Plot] Confusion matrix saved to {cm_path}")

    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print("\n=== Confusion Matrix ===")
    print(f"True Positives: {tp}, False Positives: {fp}")
    print(f"False Negatives: {fn}, True Negatives: {tn}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def find_optimal_threshold(y_true, y_pred_probs):
    """Find optimal classification threshold based on F1 score"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\n[Threshold] Optimal threshold: {best_threshold:.2f} (F1 = {best_f1:.4f})")

    return best_threshold


def save_predictions(ids, y_true, y_pred_probs, output_dir):
    """Save detailed prediction results to CSV"""
    results_df = pd.DataFrame(
        {
            "id": ids,
            "true_label": y_true,
            "predicted_prob": y_pred_probs,
            "predicted_class": (y_pred_probs >= 0.5).astype(int),
        }
    )

    # Add correct/incorrect column
    results_df["correct"] = results_df["true_label"] == results_df["predicted_class"]

    # Save to CSV
    pred_path = os.path.join(output_dir, "predictions.csv")
    results_df.to_csv(pred_path, index=False)
    print(f"[Results] Detailed predictions saved to {pred_path}")

    # Identify most confident correct and incorrect predictions
    top_correct = results_df[results_df["correct"]].nlargest(10, "predicted_prob")
    top_incorrect = results_df[~results_df["correct"]].nlargest(10, "predicted_prob")

    print("\n=== Most Confident Correct Predictions ===")
    print(top_correct[["id", "true_label", "predicted_prob"]].to_string(index=False))

    print("\n=== Most Confident Incorrect Predictions ===")
    print(top_incorrect[["id", "true_label", "predicted_prob"]].to_string(index=False))

    return results_df


def plot_pr_curve(y_true, y_pred_probs, output_dir):
    """Create and save Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=2, label=f"PR curve (AP = {ap:.3f})")
    plt.axhline(
        y=sum(y_true) / len(y_true),
        color="r",
        linestyle="--",
        label=f"Baseline (No skill): {sum(y_true)/len(y_true):.3f}",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Precision-Recall Curve", fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save plot
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] PR curve saved to {pr_path}")

    return ap


def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up device
    device = setup_device()

    # Load data
    X_seq_test, X_struct_test, y_test, ids_test = load_data(args.data_dir)
    if X_seq_test is None:
        return

    # Get the structural dimension for initializing the model
    struct_dim = X_struct_test.shape[1]

    # Load model
    model = load_model(args.model_path, struct_dim, device)
    if model is None:
        return

    # Get predictions
    print("\n[Eval] Generating predictions...")
    y_true, y_pred_probs, _ = get_predictions(
        model, X_seq_test, X_struct_test, y_test, device, args.batch_size
    )

    # Calculate and save metrics
    print("\n=== Model Evaluation ===")

    # Plot ROC curve
    auc = plot_roc_curve(y_true, y_pred_probs, args.output_dir)
    print(f"[Metrics] ROC AUC: {auc:.4f}")

    # Plot PR curve
    ap = plot_pr_curve(y_true, y_pred_probs, args.output_dir)
    print(f"[Metrics] Average Precision: {ap:.4f}")

    # Find optimal threshold
    if args.find_threshold:
        threshold = find_optimal_threshold(y_true, y_pred_probs)
    else:
        threshold = 0.5

    # Plot confusion matrix with the determined threshold
    cm_metrics = plot_confusion_matrix(y_true, y_pred_probs, threshold, args.output_dir)

    # Calculate MCC (Matthews Correlation Coefficient)
    y_pred = (y_pred_probs >= threshold).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Print comprehensive metrics
    print("\n=== Classification Metrics ===")
    print(f"Accuracy:    {cm_metrics['accuracy']:.4f}")
    print(f"Precision:   {cm_metrics['precision']:.4f}")
    print(f"Recall:      {cm_metrics['recall']:.4f}")
    print(f"Specificity: {cm_metrics['specificity']:.4f}")
    print(f"F1 Score:    {cm_metrics['f1']:.4f}")
    print(f"MCC:         {mcc:.4f}")

    # Generate classification report
    print("\n=== Classification Report ===")
    print(
        classification_report(y_true, y_pred, target_names=["Non-Allergen", "Allergen"])
    )

    # Save detailed predictions to CSV
    save_predictions(ids_test, y_true, y_pred_probs, args.output_dir)

    # Save all metrics to CSV
    metrics_df = pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "mcc",
                "auc",
                "avg_precision",
            ],
            "value": [
                cm_metrics["accuracy"],
                cm_metrics["precision"],
                cm_metrics["recall"],
                cm_metrics["specificity"],
                cm_metrics["f1"],
                mcc,
                auc,
                ap,
            ],
        }
    )
    metrics_path = os.path.join(args.output_dir, "detailed_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[Results] Detailed metrics saved to {metrics_path}")

    print("\n[Completed] Evaluation finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate allergen CNN model with detailed metrics"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with preprocessed .npy data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--find-threshold",
        action="store_true",
        help="Find optimal classification threshold based on F1",
    )
    args = parser.parse_args()

    main(args)
