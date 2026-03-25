"""
Training loop for the Player Similarity Model.

Handles:
  - Multi-objective training (action + outcome + contrastive).
  - Validation with early stopping.
  - Checkpointing.
  - Gradient clipping.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainingConfig, POSITION_IDX_TO_GROUP
from ..phase4_model.model import PlayerSimilarityModel
from .losses import CombinedLoss

_POS_TO_GROUP_T = torch.tensor(POSITION_IDX_TO_GROUP, dtype=torch.long)


class Trainer:
    """
    Trains the PlayerSimilarityModel with combined objectives.
    """

    def __init__(
        self,
        model: PlayerSimilarityModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                # MPS is available on Apple Silicon but PyG hetero-graph
                # workloads with many small kernels run faster on CPU.
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5,
        )

        self.criterion = CombinedLoss(
            lambda_outcome=config.lambda_outcome,
            lambda_contrast=config.lambda_contrast,
            lambda_pooled_contrast=config.lambda_pooled_contrast,
            contrastive_temperature=config.contrastive_temperature,
        ).to(self.device)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, list] = {
            "train_total": [],
            "train_action": [],
            "train_outcome": [],
            "train_contrastive": [],
            "train_pooled_uniform": [],
            "val_total": [],
            "val_supervised": [],
            "val_action": [],
            "val_outcome": [],
            "val_contrastive": [],
            "val_pooled_uniform": [],
            "learning_rate": [],
        }

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Schedule helpers ───────────────────────────────────────────────

    @staticmethod
    def _cosine_anneal(start: float, end: float, progress: float) -> float:
        """Cosine annealing from *start* to *end* as *progress* goes 0 -> 1."""
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

    def _apply_schedule(self, epoch: int) -> None:
        """Apply temperature / uniformity-weight annealing at epoch start."""
        cfg = self.config
        n = max(cfg.num_epochs - 1, 1)
        progress = epoch / n

        temp = cfg.contrastive_temperature
        lam_pool = cfg.lambda_pooled_contrast

        if cfg.anneal_temperature:
            temp = self._cosine_anneal(cfg.temperature_start, cfg.temperature_end, progress)
        if cfg.anneal_pooled_weight:
            lam_pool = self._cosine_anneal(cfg.pooled_weight_start, cfg.pooled_weight_end, progress)

        if cfg.anneal_temperature or cfg.anneal_pooled_weight:
            self.criterion.set_schedule(temp, lam_pool)
            if epoch % 20 == 0:
                print(f"  [schedule] temp={temp:.4f}, lambda_pooled={lam_pool:.4f}")

    # ── Training loop ────────────────────────────────────────────────

    def train(self, resume_from: str = None) -> Dict[str, list]:
        start_epoch = 0
        last_epoch = -1
        last_val_loss = float("inf")

        if resume_from:
            last_epoch, last_val_loss = self.load_checkpoint(resume_from)
            start_epoch = last_epoch + 1
            print(f"Resumed from {resume_from} (epoch {start_epoch}), "
                  f"best_val_loss={self.best_val_loss:.4f}, "
                  f"patience={self.patience_counter}/{self.config.patience}")

        if start_epoch >= self.config.num_epochs:
            print(f"\nAlready completed {start_epoch} epochs "
                  f"(num_epochs={self.config.num_epochs}). Nothing to do.")
            return self.history

        print(f"Training on device: {self.device}")
        print(f"Epochs: {self.config.num_epochs}, Batch size: {self.config.batch_size}")
        print(f"LR: {self.config.learning_rate}")

        for epoch in range(start_epoch, self.config.num_epochs):
            self._apply_schedule(epoch)

            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'=' * 50}")

            train_losses = self._train_epoch()
            val_losses = self._validate()

            # Supervised-only metric: immune to annealing schedule changes
            val_supervised = (val_losses["action"]
                              + self.config.lambda_outcome * val_losses["outcome"])

            last_epoch = epoch
            last_val_loss = val_supervised

            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_supervised)

            self.history["train_total"].append(train_losses["total"])
            self.history["train_action"].append(train_losses["action"])
            self.history["train_outcome"].append(train_losses["outcome"])
            self.history["train_contrastive"].append(train_losses["contrastive"])
            self.history["train_pooled_uniform"].append(train_losses["pooled_uniform"])
            self.history["val_total"].append(val_losses["total"])
            self.history["val_supervised"].append(val_supervised)
            self.history["val_action"].append(val_losses["action"])
            self.history["val_outcome"].append(val_losses["outcome"])
            self.history["val_contrastive"].append(val_losses["contrastive"])
            self.history["val_pooled_uniform"].append(val_losses["pooled_uniform"])
            self.history["learning_rate"].append(lr)

            print(f"Train — total: {train_losses['total']:.4f}  "
                  f"action: {train_losses['action']:.4f}  "
                  f"outcome: {train_losses['outcome']:.4f}  "
                  f"contrast: {train_losses['contrastive']:.4f}  "
                  f"uniform: {train_losses['pooled_uniform']:.4f}")
            print(f"Val   — supervised: {val_supervised:.4f}  "
                  f"action: {val_losses['action']:.4f}  "
                  f"outcome: {val_losses['outcome']:.4f}  "
                  f"contrast: {val_losses['contrastive']:.4f}  "
                  f"uniform: {val_losses['pooled_uniform']:.4f}  "
                  f"LR: {lr:.6f}")

            if val_supervised < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_supervised
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch, val_supervised)
                print("  [New best model saved]")
            else:
                self.patience_counter += 1
                print(f"  [No improvement for {self.patience_counter} epochs]")

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", epoch, val_supervised)

            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping after {epoch + 1} epochs.")
                break

        self._save_checkpoint("final_model.pt", last_epoch, last_val_loss)
        self._save_history()
        return self.history

    # ── Epoch helpers ────────────────────────────────────────────────

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        accum = {"total": 0.0, "action": 0.0, "outcome": 0.0, "contrastive": 0.0, "pooled_uniform": 0.0}
        n_batches = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            data = batch["data"].to(self.device)
            action_targets = {k: v.to(self.device) for k, v in batch["action_targets"].items()}
            outcome_targets = batch["outcome_targets"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(data)

            # Outcome: mean-pool event outcomes per possession then
            # expand to match the per-possession outcome_targets.
            # For simplicity, use the mean of per-event outcome logits.
            outcome_logits_per_event = outputs["outcome"]  # (E_total, 2)

            # Build per-possession outcome logits via scatter-mean
            # using the batch vector for the "event" node type.
            batch_vec = data["event"].batch  # (E_total,)
            n_poss = outcome_targets.shape[0]
            outcome_logits = torch.zeros(n_poss, 2, device=self.device)
            counts = torch.zeros(n_poss, 1, device=self.device)
            outcome_logits.scatter_add_(0, batch_vec.unsqueeze(-1).expand(-1, 2), outcome_logits_per_event)
            counts.scatter_add_(0, batch_vec.unsqueeze(-1), torch.ones_like(batch_vec, dtype=torch.float32).unsqueeze(-1))
            counts = counts.clamp(min=1.0)
            outcome_logits = outcome_logits / counts

            # Contrastive loss on pre-pooling h_player with hard negatives.
            # h_player has one entry per player node in the batch; the same
            # player_id can appear in multiple possessions, giving positive
            # pairs.  Position groups restrict negatives to same-role players.
            player_node_pids = data["player"].player_node_pids if hasattr(data["player"], "player_node_pids") else None

            cont_emb = None
            cont_pids = None
            cont_pos_groups = None
            if player_node_pids is not None:
                actor_mask = player_node_pids >= 0
                if actor_mask.any():
                    cont_emb = outputs["h_player"][actor_mask]
                    cont_pids = player_node_pids[actor_mask]
                    pos_group_map = _POS_TO_GROUP_T.to(self.device)
                    n_pos = pos_group_map.shape[0]
                    actor_pos_idx = data["player"].x[actor_mask, 0].long().clamp(0, n_pos - 1)
                    cont_pos_groups = pos_group_map[actor_pos_idx]

            pooled_emb = outputs.get("pooled_z_p")

            losses = self.criterion(
                preds={
                    "action_type": outputs["action_type"],
                    "angle_bin": outputs["angle_bin"],
                    "length_bin": outputs["length_bin"],
                },
                action_targets=action_targets,
                outcome_logits=outcome_logits,
                outcome_targets=outcome_targets,
                contrastive_embeddings=cont_emb,
                contrastive_player_ids=cont_pids,
                contrastive_position_groups=cont_pos_groups,
                pooled_embeddings=pooled_emb,
            )

            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in accum:
                accum[k] += losses[k].item()
            n_batches += 1

        n = max(n_batches, 1)
        return {k: v / n for k, v in accum.items()}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        accum = {"total": 0.0, "action": 0.0, "outcome": 0.0, "contrastive": 0.0, "pooled_uniform": 0.0}
        n_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            data = batch["data"].to(self.device)
            action_targets = {k: v.to(self.device) for k, v in batch["action_targets"].items()}
            outcome_targets = batch["outcome_targets"].to(self.device)

            outputs = self.model(data)

            batch_vec = data["event"].batch
            n_poss = outcome_targets.shape[0]
            outcome_logits = torch.zeros(n_poss, 2, device=self.device)
            counts = torch.zeros(n_poss, 1, device=self.device)
            outcome_logits.scatter_add_(0, batch_vec.unsqueeze(-1).expand(-1, 2), outputs["outcome"])
            counts.scatter_add_(0, batch_vec.unsqueeze(-1), torch.ones_like(batch_vec, dtype=torch.float32).unsqueeze(-1))
            counts = counts.clamp(min=1.0)
            outcome_logits = outcome_logits / counts

            player_node_pids = data["player"].player_node_pids if hasattr(data["player"], "player_node_pids") else None
            cont_emb = None
            cont_pids = None
            cont_pos_groups = None
            if player_node_pids is not None:
                actor_mask = player_node_pids >= 0
                if actor_mask.any():
                    cont_emb = outputs["h_player"][actor_mask]
                    cont_pids = player_node_pids[actor_mask]
                    pos_group_map = _POS_TO_GROUP_T.to(self.device)
                    n_pos = pos_group_map.shape[0]
                    actor_pos_idx = data["player"].x[actor_mask, 0].long().clamp(0, n_pos - 1)
                    cont_pos_groups = pos_group_map[actor_pos_idx]

            pooled_emb = outputs.get("pooled_z_p")

            losses = self.criterion(
                preds={
                    "action_type": outputs["action_type"],
                    "angle_bin": outputs["angle_bin"],
                    "length_bin": outputs["length_bin"],
                },
                action_targets=action_targets,
                outcome_logits=outcome_logits,
                outcome_targets=outcome_targets,
                contrastive_embeddings=cont_emb,
                contrastive_player_ids=cont_pids,
                contrastive_position_groups=cont_pos_groups,
                pooled_embeddings=pooled_emb,
            )
            for k in accum:
                accum[k] += losses[k].item()
            n_batches += 1

        n = max(n_batches, 1)
        return {k: v / n for k, v in accum.items()}

    # ── Persistence ──────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "history": self.history,
        }, path)

    def load_checkpoint(self, filename: str) -> Tuple[int, float]:
        path = self.checkpoint_dir / filename
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.patience_counter = ckpt.get("patience_counter", 0)
        saved_history = ckpt.get("history")
        if saved_history and isinstance(saved_history, dict):
            self.history = saved_history
        return ckpt["epoch"], ckpt["val_loss"]

    def _save_history(self):
        path = self.checkpoint_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
