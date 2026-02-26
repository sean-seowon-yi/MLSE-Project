"""
Test-set evaluation for the Player Similarity Model.

Runs the trained model on the held-out test split and computes:
  - Action heads: accuracy, macro F1, confusion matrices (action type, angle, length).
  - Outcome head: accuracy, BCE, AUC-ROC for ends_in_shot and ends_in_goal.

Saves metrics to JSON and visualizations (confusion matrices, ROC curves) to disk.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from ..config import EVENT_TYPES, ModelConfig, TrainingConfig
from ..phase4_model.model import PlayerSimilarityModel
from .dataset import PossessionGraphDataset, collate_fn, train_val_test_split


def _aggregate_outcome_logits(
    outcome_logits: torch.Tensor,
    batch_vec: torch.Tensor,
    n_poss: int,
    device: torch.device,
) -> torch.Tensor:
    """Scatter-mean per possession (same logic as trainer)."""
    outcome_logits = outcome_logits.to(device)
    batch_vec = batch_vec.to(device)
    out = torch.zeros(n_poss, 2, device=device)
    counts = torch.zeros(n_poss, 1, device=device)
    out.scatter_add_(0, batch_vec.unsqueeze(-1).expand(-1, 2), outcome_logits)
    counts.scatter_add_(
        0, batch_vec.unsqueeze(-1),
        torch.ones_like(batch_vec, dtype=torch.float32, device=device).unsqueeze(-1),
    )
    counts = counts.clamp(min=1.0)
    logits = (out / counts).cpu()
    probs = torch.sigmoid(logits).numpy()
    return probs


def run_test_evaluation(
    model: PlayerSimilarityModel,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[
    np.ndarray, np.ndarray,  # action_type pred, true
    np.ndarray, np.ndarray,  # angle_bin pred, true
    np.ndarray, np.ndarray,  # length_bin pred, true
    np.ndarray, np.ndarray,  # outcome pred (n_poss, 2), true (n_poss, 2)
]:
    """Run model on test loader and collect all predictions and targets."""
    model.eval()
    action_type_pred_list: List[np.ndarray] = []
    action_type_true_list: List[np.ndarray] = []
    angle_bin_pred_list: List[np.ndarray] = []
    angle_bin_true_list: List[np.ndarray] = []
    length_bin_pred_list: List[np.ndarray] = []
    length_bin_true_list: List[np.ndarray] = []
    outcome_pred_list: List[np.ndarray] = []
    outcome_true_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test evaluation"):
            data = batch["data"].to(device)
            action_targets = {k: v.cpu().numpy() for k, v in batch["action_targets"].items()}
            outcome_targets = batch["outcome_targets"].cpu().numpy()

            outputs = model(data)

            # Per-event predictions (argmax)
            action_type_pred_list.append(outputs["action_type"].argmax(dim=-1).cpu().numpy())
            action_type_true_list.append(action_targets["action_type"])
            angle_bin_pred_list.append(outputs["angle_bin"].argmax(dim=-1).cpu().numpy())
            angle_bin_true_list.append(action_targets["angle_bin"])
            length_bin_pred_list.append(outputs["length_bin"].argmax(dim=-1).cpu().numpy())
            length_bin_true_list.append(action_targets["length_bin"])

            # Per-possession outcome (scatter-mean)
            batch_vec = data["event"].batch
            n_poss = outcome_targets.shape[0]
            out_probs = _aggregate_outcome_logits(
                outputs["outcome"], batch_vec, n_poss, device
            )
            outcome_pred_list.append(out_probs)
            outcome_true_list.append(outcome_targets)

    action_type_pred = np.concatenate(action_type_pred_list, axis=0)
    action_type_true = np.concatenate(action_type_true_list, axis=0)
    angle_bin_pred = np.concatenate(angle_bin_pred_list, axis=0)
    angle_bin_true = np.concatenate(angle_bin_true_list, axis=0)
    length_bin_pred = np.concatenate(length_bin_pred_list, axis=0)
    length_bin_true = np.concatenate(length_bin_true_list, axis=0)
    outcome_pred = np.concatenate(outcome_pred_list, axis=0)
    outcome_true = np.concatenate(outcome_true_list, axis=0)

    return (
        action_type_pred, action_type_true,
        angle_bin_pred, angle_bin_true,
        length_bin_pred, length_bin_true,
        outcome_pred, outcome_true,
    )


def compute_metrics(
    action_type_pred: np.ndarray,
    action_type_true: np.ndarray,
    angle_bin_pred: np.ndarray,
    angle_bin_true: np.ndarray,
    length_bin_pred: np.ndarray,
    length_bin_true: np.ndarray,
    outcome_pred: np.ndarray,
    outcome_true: np.ndarray,
) -> Dict:
    """Compute accuracy, macro F1, and outcome metrics. Use sklearn."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
    )

    metrics: Dict = {}

    # Action type
    metrics["action_type"] = {
        "accuracy": float(accuracy_score(action_type_true, action_type_pred)),
        "macro_f1": float(f1_score(action_type_true, action_type_pred, average="macro", zero_division=0)),
    }
    metrics["action_type"]["confusion_matrix"] = confusion_matrix(
        action_type_true, action_type_pred, labels=np.arange(len(EVENT_TYPES))
    ).tolist()

    # Angle bin
    metrics["angle_bin"] = {
        "accuracy": float(accuracy_score(angle_bin_true, angle_bin_pred)),
        "macro_f1": float(f1_score(angle_bin_true, angle_bin_pred, average="macro", zero_division=0)),
    }

    # Length bin
    metrics["length_bin"] = {
        "accuracy": float(accuracy_score(length_bin_true, length_bin_pred)),
        "macro_f1": float(f1_score(length_bin_true, length_bin_pred, average="macro", zero_division=0)),
    }

    # Outcome: ends_in_shot (col 0), ends_in_goal (col 1)
    pred_shot = outcome_pred[:, 0]
    pred_goal = outcome_pred[:, 1]
    true_shot = outcome_true[:, 0]
    true_goal = outcome_true[:, 1]

    def binary_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        y_pred = (y_pred_proba >= 0.5).astype(np.float64)
        return float(accuracy_score(y_true, y_pred))

    metrics["outcome"] = {
        "ends_in_shot": {
            "accuracy": binary_accuracy(true_shot, pred_shot),
            "bce": float(np.mean(
                -true_shot * np.log(np.clip(pred_shot, 1e-7, 1 - 1e-7))
                - (1 - true_shot) * np.log(np.clip(1 - pred_shot, 1e-7, 1 - 1e-7))
            )),
            "auc_roc": float(roc_auc_score(true_shot, pred_shot)) if np.unique(true_shot).size > 1 else 0.0,
        },
        "ends_in_goal": {
            "accuracy": binary_accuracy(true_goal, pred_goal),
            "bce": float(np.mean(
                -true_goal * np.log(np.clip(pred_goal, 1e-7, 1 - 1e-7))
                - (1 - true_goal) * np.log(np.clip(1 - pred_goal, 1e-7, 1 - 1e-7))
            )),
            "auc_roc": float(roc_auc_score(true_goal, pred_goal)) if np.unique(true_goal).size > 1 else 0.0,
        },
    }

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """Save confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_outcome_roc(
    outcome_pred: np.ndarray,
    outcome_true: np.ndarray,
    out_path: Path,
) -> None:
    """Plot ROC curves for ends_in_shot and ends_in_goal."""
    from sklearn.metrics import roc_curve

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(["ends_in_shot", "ends_in_goal"]):
        y_true = outcome_true[:, i]
        y_score = outcome_pred[:, i]
        if np.unique(y_true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test set: Outcome ROC curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_visualizations(
    action_type_true: np.ndarray,
    action_type_pred: np.ndarray,
    angle_bin_true: np.ndarray,
    angle_bin_pred: np.ndarray,
    length_bin_true: np.ndarray,
    length_bin_pred: np.ndarray,
    outcome_pred: np.ndarray,
    outcome_true: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate and save all evaluation plots."""
    from sklearn.metrics import confusion_matrix

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Action type confusion matrix
    cm_at = confusion_matrix(
        action_type_true, action_type_pred, labels=np.arange(len(EVENT_TYPES))
    )
    plot_confusion_matrix(
        cm_at,
        EVENT_TYPES,
        "Test set: Action type confusion matrix",
        output_dir / "confusion_matrix_action_type.png",
        figsize=(10, 8),
    )

    # Angle bin (labels 0..8: 8 sectors + no-angle)
    n_angle_bins = 9
    angle_labels = [f"Sector {i}" if i < 8 else "No angle" for i in range(n_angle_bins)]
    cm_angle = confusion_matrix(
        angle_bin_true, angle_bin_pred, labels=np.arange(n_angle_bins)
    )
    plot_confusion_matrix(
        cm_angle,
        angle_labels,
        "Test set: Angle bin confusion matrix",
        output_dir / "confusion_matrix_angle_bin.png",
        figsize=(8, 6),
    )

    # Length bin (5 bins)
    length_labels = ["Very short", "Short", "Medium", "Long", "Very long"]
    cm_len = confusion_matrix(
        length_bin_true, length_bin_pred, labels=np.arange(5)
    )
    plot_confusion_matrix(
        cm_len,
        length_labels,
        "Test set: Length bin confusion matrix",
        output_dir / "confusion_matrix_length_bin.png",
        figsize=(6, 5),
    )

    # Outcome ROC
    plot_outcome_roc(outcome_pred, outcome_true, output_dir / "outcome_roc.png")


def evaluate(
    model: PlayerSimilarityModel,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    save_plots: bool = True,
) -> Dict:
    """
    Run full test evaluation: collect preds/targets, compute metrics, save JSON and plots.

    Returns the metrics dict (also written to output_dir / "test_metrics.json").
    """
    (
        action_type_pred, action_type_true,
        angle_bin_pred, angle_bin_true,
        length_bin_pred, length_bin_true,
        outcome_pred, outcome_true,
    ) = run_test_evaluation(model, test_loader, device)

    metrics = compute_metrics(
        action_type_pred, action_type_true,
        angle_bin_pred, angle_bin_true,
        length_bin_pred, length_bin_true,
        outcome_pred, outcome_true,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if save_plots:
        save_visualizations(
            action_type_true, action_type_pred,
            angle_bin_true, angle_bin_pred,
            length_bin_true, length_bin_pred,
            outcome_pred, outcome_true,
            output_dir,
        )

    return metrics
